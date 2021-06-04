import abc
import logging
from typing import Optional

import numpy as np
import torch

from rltoolkit import config
from rltoolkit.utils import standardize_and_clip

logger = logging.getLogger(__name__)


class MemoryMeta:
    def __init__(
        self,
        obs_mean: Optional[torch.tensor] = None,
        obs_std: Optional[torch.tensor] = None,
        device: Optional[torch.device] = None,
        min_max_denormalize: bool = False,
        max_obs: Optional[torch.tensor] = None,
        min_obs: Optional[torch.tensor] = None,
    ):
        """Metaclass for extraction of normalized values from memory
        """
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self._obs = None
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        self.min_max_denormalize = min_max_denormalize
        self.max_obs = max_obs
        self.min_obs = min_obs

    @property
    @abc.abstractmethod
    def obs(self):
        raise NotImplementedError

    @property
    def next_obs(self):
        raise NotImplementedError

    @property
    def norm_obs(self):
        return self.normalize(self.obs)

    @property
    def norm_next_obs(self):
        return self.normalize(self.next_obs)

    @property
    def obs_mean(self):
        return self._obs_mean

    @property
    def obs_std(self):
        return self._obs_std

    @obs_mean.setter
    def obs_mean(self, value: Optional[torch.tensor]):
        if value is None:
            self._obs_mean = value
        else:
            self._obs_mean = value.to(self.device)

    @obs_std.setter
    def obs_std(self, value: Optional[torch.tensor]):
        if value is None:
            self._obs_std = value
        else:
            self._obs_std = value.to(self.device)

    def normalize(self, obs: torch.tensor, ob_idx : list = None) -> torch.tensor:
        both_nones = self.min_obs is None and self.max_obs is None
        if self.min_max_denormalize:
            if both_nones:
                return obs

            if ob_idx is None:
                mask = self.min_obs != self.max_obs
                masked_min = self.min_obs[mask]
                masked_max = self.max_obs[mask]
            else:
                mask = self.min_obs[ob_idx] != self.max_obs[ob_idx]
                masked_min = self.min_obs[ob_idx][mask]
                masked_max = self.max_obs[ob_idx][mask]

            masked_obs = obs[:, mask]
            mean = (masked_max + masked_min) / 2
            max_delta = (masked_max - masked_min) / 2            
            masked_obs = (masked_obs - mean) / (max_delta + 1e-8)
            obs[:, mask] = masked_obs
        else:
            if both_nones:
                return obs
            obs = standardize_and_clip(obs, self.obs_mean, self.obs_std)
        return obs

    def denormalize(self, obs: torch.tensor, ob_idx: list = None) -> torch.tensor:
        """
        Denormalize RL action in AcM setup. If self.min_max_denormalize is True input in
            range (-1.1, 1.1) is expected (it can be initially greater due to normal
            dist in the agent)

        Args:
            obs (torch.tensor): observations to denormalize (or actions before ACM)
            ob_idx(list, optional): indexes of observations included in obs

        Returns:
            torch.tensor: denormalized observations (or actions before ACM)
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
            was_numpy = True
        else:
            was_numpy = False

        if self.min_max_denormalize:
            if self.max_obs is None and self.min_obs is None:
                logger.warning(
                    "max_obs and min_obs are Nones and will be calculated on the go "
                    "this time."
                )
                max_obs = torch.from_numpy(np.percentile(self.obs, 99, axis=0)).float()
                min_obs = torch.from_numpy(np.percentile(self.obs, 1, axis=0)).float()
            else:
                max_obs = self.max_obs
                min_obs = self.min_obs
            
            if ob_idx is not None:
                max_obs = max_obs[ob_idx]
                min_obs = min_obs[ob_idx]

            mean = (max_obs + min_obs) / 2
            max_delta = (max_obs - min_obs) / 2
            obs = mean + obs * max_delta
        else:
            obs = (self.obs_std + 1e-8) * obs + self.obs_mean

        if was_numpy:
            obs = obs.numpy()
        return obs


class Memory(MemoryMeta):
    def __init__(self, alpha: float = config.NORM_ALPHA, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs = []
        self._next_obs_idx = []
        self._actions = []
        self._action_logprobs = []
        self._rewards = []
        self._done = []
        self._end = []
        self._new_rollout_idx = []  # For obs skip previous, for next_obs skip this
        self.current_len = 0
        if alpha is not None:
            assert 0 <= alpha <= 1, "Alpha needs to be in <0, 1> range."
        self.alpha = alpha

    @property
    def obs(self):
        obs = []
        rollout = 0
        for i in range(self.current_len):
            if i != self._new_rollout_idx[rollout] - 1:
                obs.append(self._obs[i])
            else:
                if rollout < len(self._new_rollout_idx) - 1:
                    rollout += 1
        return torch.cat(obs)

    @property
    def next_obs(self):
        obs = []
        rollout = 0
        for i in range(1, self.current_len):
            if i != self._new_rollout_idx[rollout]:
                obs.append(self._obs[i])
            else:
                if rollout < len(self._new_rollout_idx) - 1:
                    rollout += 1
        return torch.cat(obs)

    @property
    def norm_obs(self):
        return self.normalize(self.obs)

    @property
    def norm_next_obs(self):
        return self.normalize(self.next_obs)

    @property
    def actions(self):
        return self._actions

    @property
    def action_logprobs(self):
        return self._action_logprobs

    @property
    def rewards(self):
        return self._rewards

    @property
    def done(self):
        return self._done

    @property
    def end(self):
        return self._end

    @property
    def average_returns_per_rollout(self):
        return sum(self.returns_rollouts) / self.rollouts_no

    @property
    def returns_rollouts(self) -> np.array:
        returns = []
        return_ = 0

        for i, end in enumerate(self.end):
            return_ += self.rewards[i]
            if end:
                returns.append(return_)
                return_ = 0
        return np.array(returns)

    @property
    def rollouts_no(self) -> int:
        return sum(self._end)

    def add_obs(self, obs):
        self._obs.append(obs)
        obs_idx = len(self._obs) - 1
        self.current_len += 1
        return obs_idx

    def add_timestep(
        self, obs_idx, next_obs_idx, action, action_logprobs, reward, done, end
    ):
        self._next_obs_idx.append(next_obs_idx)
        self._actions.append(action)
        self._action_logprobs.append(action_logprobs)
        self._rewards.append(reward)
        self._done.append(done)
        self._end.append(end)

    def __len__(self):
        return len(self._next_obs_idx)

    def __getitem__(self, obs_idx):
        next_obs_idx = self._next_obs_idx[obs_idx]
        return (
            self._obs[obs_idx],
            self._obs[next_obs_idx],
            self._actions[obs_idx],
            self._action_logprobs[obs_idx],
            self._rewards[obs_idx],
            self._done[obs_idx],
        )

    def new_rollout(self):
        if len(self._obs) > 0:
            self._new_rollout_idx.append(self.current_len)

    def end_rollout(self):
        self._new_rollout_idx.append(self.current_len)

    def add_rollout(self, _obs, actions, action_logprobs, rewards, dones):
        len_before = self.__len__()
        self._obs += _obs
        self._actions += actions
        self._action_logprobs += action_logprobs
        self._rewards += rewards
        self._done += dones
        self._end += dones
        self._end[-1] = True

        new_next_idx = list(range(len_before + 1, len(self._obs)))
        self._next_obs_idx.append(new_next_idx)
        self.current_len += len(_obs)
        self.end_rollout()

    def add_buffers_list(self, buffer_list):
        for new_buffer in buffer_list:
            self._obs += new_buffer._obs
            self._actions += new_buffer._actions
            self._action_logprobs += new_buffer._action_logprobs
            self._rewards += new_buffer._rewards
            self._done += new_buffer._done
            self._end += new_buffer._end

            last_idx = 0 if len(self._next_obs_idx) == 0 else self._next_obs_idx[-1]
            for i in new_buffer._next_obs_idx:
                self._next_obs_idx.append(last_idx + i + 1)

    def update_obs_mean_std(self):
        obs = torch.cat(self._obs).squeeze()
        mean = obs.mean(axis=0).to(self.device)
        std = obs.std(axis=0).to(self.device)
        if self.obs_std is None and self.obs_mean is None:
            self.obs_mean = mean
            self.obs_std = std
        else:
            self.obs_mean = (1 - self.alpha) * mean + self.alpha * self.obs_mean
            self.obs_std = (1 - self.alpha) * std + self.alpha * self.obs_std

        cur_max = torch.from_numpy(np.percentile(self.obs, 99, axis=0)).float()
        cur_min = torch.from_numpy(np.percentile(self.obs, 1, axis=0)).float()

        if self.max_obs is None or self.min_obs is None:
            self.max_obs = cur_max
            self.min_obs = cur_min
        else:
            self.max_obs = torch.max(cur_max, self.max_obs)
            self.min_obs = torch.min(cur_min, self.min_obs)


class MemoryAcM(Memory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions_acm = []

    def add_acm_action(self, acm_action):
        self.actions_acm.append(acm_action)

    def add_rollout(self, _obs, actions, action_logprobs, rewards, dones, actions_acm):
        super().add_rollout(_obs, actions, action_logprobs, rewards, dones)
        self.actions_acm += actions_acm
