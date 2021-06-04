import numpy as np
import torch

from rltoolkit.buffer.memory import Memory, MemoryAcM, MemoryMeta


class MetaReplayBuffer(MemoryMeta):
    def __init__(
        self,
        size: int,
        obs_shape,
        obs_norm: bool = False,
        dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.size = size
        self.obs_shape = obs_shape
        self.obs_norm = obs_norm
        self.dtype = dtype

        self.obs_idx = None
        self.ts_idx = None
        self.current_len = None
        self.reset_idx()

        self._obs = np.empty([self.size, self.obs_shape])
        self._obs_idx = np.empty(self.size, dtype=np.int)
        self._next_obs_idx = np.empty(self.size, dtype=np.int)

    def reset_idx(self):
        self.obs_idx = 0
        self.ts_idx = 0
        self.current_len = 0

    @property
    def obs(self):
        return self._obs[self._obs_idx[: self.current_len]]

    @property
    def norm_obs(self):
        return self.normalize(self.obs)

    @property
    def next_obs(self):
        return self._obs[self._next_obs_idx[: self.current_len]]

    @property
    def norm_next_obs(self):
        return self.normalize(self.next_obs)

    def __len__(self):
        return self.current_len

    def add_obs(self, obs: torch.Tensor):
        self._obs[self.obs_idx] = obs.cpu()
        obs_idx = self.obs_idx
        self.obs_idx = (self.obs_idx + 1) % self.size
        return obs_idx

    def addition(*args, **kwargs):
        pass

    def add_timestep(self, obs_idx: int, next_obs_idx: int, *args, **kwargs):
        self._obs_idx[self.ts_idx] = obs_idx
        self._next_obs_idx[self.ts_idx] = next_obs_idx
        self.addition(*args, **kwargs)

        if next_obs_idx < self.ts_idx:
            self.current_len = self.ts_idx + 1
            self.ts_idx = 0
        else:
            self.ts_idx += 1
        self.current_len = max(self.ts_idx, self.current_len)

    def normalize(self, obs: torch.tensor, force=False) -> torch.tensor:
        if self.obs_norm or force:
            return super().normalize(obs)
        else:
            return obs

    def update_obs_mean_std(self):
        if len(self.obs) > 10:
            self.obs_mean = torch.tensor(self.obs.mean(axis=0), dtype=self.dtype)
            self.obs_std = torch.tensor(self.obs.std(axis=0), dtype=self.dtype)

            cur_max = torch.from_numpy(np.percentile(self.obs, 99, axis=0)).float()
            cur_min = torch.from_numpy(np.percentile(self.obs, 1, axis=0)).float()

            if self.max_obs is None or self.min_obs is None:
                self.max_obs = cur_max
                self.min_obs = cur_min
            else:
                self.max_obs = torch.max(cur_max, self.max_obs)
                self.min_obs = torch.min(cur_min, self.min_obs)


class ReplayBuffer(MetaReplayBuffer):
    def __init__(
        self, size: int, obs_shape, act_shape, discrete: bool = False, *args, **kwargs
    ):
        super().__init__(size, obs_shape, *args, **kwargs)
        self.discrete = discrete
        if self.discrete:
            self._actions = np.empty(self.size, dtype=np.int)
        else:
            self._actions = np.empty([self.size, act_shape])
        self._rewards = np.empty(self.size, dtype=np.float32)
        self._done = np.empty(self.size, dtype=np.bool_)
        self._end = np.empty(self.size, dtype=np.bool_)  # only for episode extraction

        if self.obs_std is None and self.obs_mean is None:
            self.obs_mean = torch.zeros(obs_shape, device=self.device)
            self.obs_std = torch.ones(obs_shape, device=self.device)

    @property
    def actions(self):
        return self._actions[: self.current_len]

    @property
    def rewards(self):
        return self._rewards[: self.current_len]

    @property
    def done(self):
        return self._done[: self.current_len]

    @property
    def end(self):
        return self._end[: self.current_len]

    def addition(self, action, rew, done, end):
        self._actions[self.ts_idx] = action.cpu()
        self._rewards[self.ts_idx] = rew
        self._done[self.ts_idx] = done
        self._end[self.ts_idx] = end

    def __getitem__(self, idx):
        assert idx < self.current_len, IndexError
        return (
            self._obs[self._obs_idx[idx]],
            self._obs[self._next_obs_idx[idx]],
            self._actions[idx],
            self._rewards[idx],
            self._done[idx],
        )

    def add_buffer(self, memory: Memory):
        raise DeprecationWarning
        obs = memory._obs
        new_rollout_idx = memory._new_rollout_idx
        i = 0
        obs_idx = self.add_obs(obs[i])
        timesteps = zip(memory.actions, memory.rewards, memory.done)
        for action, rew, done in timesteps:
            i += 1
            next_idx = self.add_obs(obs[i])
            if i in new_rollout_idx:
                i += 1
                continue
            self.add_timestep(obs_idx, next_idx, action, rew, done)
            obs_idx = next_idx

    def make_obs_memory_tensor(self, obs):
        processed = torch.tensor(obs, dtype=self.dtype, device=self.device)
        processed = processed.reshape(-1, self.obs_shape)
        return processed

    def last_end(self, idx):        
        end = self._end[idx]
        while not end:
            idx -= 1
            if idx < 0:
                idx = self.current_len - 1
            end = self._end[idx]            
        return idx

    def last_rollout(self):
        """Get elements of last full rollout:
            1. Find last end
            2. Grab elements till the next end

        Returns:
            Memory: memory with last rollout only
        """        
        i = self.last_end(self.ts_idx - 1)
        next_end = False

        obs = []
        actions = []
        rewards = []
        dones = []        
        last_obs = self.make_obs_memory_tensor(self._obs[self._next_obs_idx[i]])        
        while not next_end:
            new_obs = self.make_obs_memory_tensor(self._obs[self._obs_idx[i]])
            obs.insert(0, new_obs)
            actions.insert(0, self._actions[i])
            rewards.insert(0, self._rewards[i])
            dones.insert(0, self._end[i])
            i -= 1
            if i < 0:
                i = self.current_len - 1
            next_end = self._end[i]

        obs.append(last_obs)
        memory = Memory(
            min_max_denormalize=self.min_max_denormalize,
            min_obs=self.min_obs,
            max_obs=self.max_obs,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
        )

        memory.add_rollout(
            _obs=obs, actions=actions, action_logprobs=[], rewards=rewards, dones=dones
        )
        return memory

    def sample_batch(self, steps_per_epoch=64, device=None):
        f"""Sample batch of tensors from buffer

        Args:
            steps_per_epoch (int, optional): batch size. Defaults to { 64 }.
            device (torch.device, optional): use GPU/CPU device. Defaults to { None }.

        Returns:
            list: list of elements from buffer (obs, next_obs, actions, rewards, dones)
        """
        batch, _ = self._sample_batch(steps_per_epoch, device)
        return batch

    def _sample_batch(self, batch_size, device):
        idxs = np.random.randint(0, self.__len__(), batch_size)
        if self.discrete:
            actions_type = torch.long
        else:
            actions_type = self.dtype

        obs = torch.as_tensor(
            self._obs[self._obs_idx[idxs]], dtype=self.dtype, device=self.device
        )
        next_obs = torch.as_tensor(
            self._obs[self._next_obs_idx[idxs]], dtype=self.dtype, device=self.device
        )
        if self.obs_norm:
            obs = self.normalize(obs)
            next_obs = self.normalize(next_obs)

        batch = [
            obs,
            next_obs,
            torch.as_tensor(self._actions[idxs], dtype=actions_type),
            torch.as_tensor(self._rewards[idxs], dtype=actions_type),
            torch.as_tensor(self._done[idxs], dtype=torch.int8),
        ]

        if device:
            batch = [item.to(device) for item in batch]

        return batch, idxs


class ReplayBufferAcM(MetaReplayBuffer):
    def __init__(
        self, size: int, obs_shape, act_shape, discrete: bool = False, *args, **kwargs
    ):
        super().__init__(size, obs_shape, *args, **kwargs)
        self._next_obs_idx = np.empty(self.size, dtype=np.int)
        self.discrete = discrete
        self.act_shape = act_shape
        if self.discrete:
            self._actions_acm = np.empty(self.size, dtype=np.int)
        else:
            self._actions_acm = np.empty([self.size, self.act_shape])

    @property
    def actions_acm(self):
        return self._actions_acm[: self.current_len]

    def addition(self, acm_action):
        self._actions_acm[self.ts_idx] = acm_action

    def add_buffer(self, memory: MemoryAcM):
        obs = memory._obs
        actions_acm = memory.actions_acm
        new_rollout_idx = memory._new_rollout_idx
        i = 0
        obs_idx = self.add_obs(obs[i])
        for acm_action in actions_acm:
            i += 1
            next_idx = self.add_obs(obs[i])
            if i in new_rollout_idx:
                i += 1
                continue
            self.add_timestep(obs_idx, next_idx, acm_action)
            obs_idx = next_idx

    def sample_acm_batch(self, batch_size=64):
        return rbuffer_sample_acm(self, batch_size, self.discrete)


class BufferAcMOffPolicy(ReplayBuffer):
    def __init__(
        self,
        size: int,
        obs_shape,
        act_shape,
        acm_act_shape,
        acm_discrete: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            size,
            obs_shape=obs_shape,
            act_shape=act_shape,
            discrete=False,
            *args,
            **kwargs,
        )
        self.acm_discrete = acm_discrete
        if self.acm_discrete:
            self._actions_acm = np.empty(self.size, dtype=np.int)
        else:
            self._actions_acm = np.empty([self.size, acm_act_shape])

    @property
    def actions_acm(self):
        return self._actions_acm[: self.current_len]

    def add_acm_action(self, acm_action):
        self._actions_acm[self.ts_idx] = acm_action

    def last_rollout(self):
        """Get elements of last full rollout:
            1. Find last end
            2. Grab elements till the next end

        Returns:
            Memory: memory with last rollout only
        """        
        i = self.last_end(self.ts_idx - 1)                
        next_end = False

        obs = []
        actions = []
        rewards = []
        dones = []
        actions_acm = []
        last_obs = self.make_obs_memory_tensor(self._obs[self._next_obs_idx[i]])

        while not next_end:
            new_obs = self.make_obs_memory_tensor(self._obs[self._obs_idx[i]])
            obs.insert(0, new_obs)
            actions.insert(0, self._actions[i])
            rewards.insert(0, self._rewards[i])
            dones.insert(0, self._end[i])
            actions_acm.insert(0, self._actions_acm[i])
            i -= 1
            if i < 0:
                i = self.current_len - 1
            next_end = self._end[i]

        obs.append(last_obs)

        memory = MemoryAcM(
            min_max_denormalize=self.min_max_denormalize,
            min_obs=self.min_obs,
            max_obs=self.max_obs,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
        )

        memory.add_rollout(
            _obs=obs,
            actions=actions,
            action_logprobs=[],
            rewards=rewards,
            dones=dones,
            actions_acm=actions_acm,
        )
        return memory

    def sample_batch(self, batch_size=64, device=None):
        f"""Sample batch of tensors from buffer

        Args:
            batch_size (int, optional): batch size. Defaults to { 64 }.
            device (torch.device, optional): use GPU/CPU device. Defaults to { None }.

        Returns:
            list: list of elements from buffer (obs, next_obs, actions, rewards, dones)
        """
        batch, idxs = self._sample_batch(batch_size, device)
        acm_observations = torch.as_tensor(self._actions_acm[idxs], dtype=torch.float32)
        batch.append(acm_observations)
        return batch

    def sample_acm_batch(self, batch_size=64):
        return rbuffer_sample_acm(self, batch_size, self.acm_discrete)


def rbuffer_sample_acm(
    replay_buffer: MetaReplayBuffer, batch_size: int, discrete: bool
):
    """Sample batch of tensors from buffer

    Args:
        replay_buffer (MetaReplayBuffer): replay buffer to sample.
        batch_size (int): batch size.
        discrete (bool): discrete action samples or not.

    Returns:
        list: list of elements from buffer (obs, next_obs, actions)
    """
    dtype = replay_buffer.dtype
    idxs = np.random.randint(0, len(replay_buffer), batch_size)
    if discrete:
        acm_actions_type = torch.long
    else:
        acm_actions_type = dtype

    obs = torch.as_tensor(replay_buffer._obs[replay_buffer._obs_idx[idxs]], dtype=dtype)
    next_obs = torch.as_tensor(
        replay_buffer._obs[replay_buffer._next_obs_idx[idxs]], dtype=dtype
    )
    actions = torch.as_tensor(replay_buffer._actions_acm[idxs], dtype=acm_actions_type)

    return [obs, next_obs, actions]
