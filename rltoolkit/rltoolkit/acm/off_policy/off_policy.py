import torch

from rltoolkit import config
from rltoolkit.acm import AcMTrainer
from rltoolkit.buffer import BufferAcMOffPolicy, MemoryAcM



class AcMOffPolicy(AcMTrainer):
    def __init__(
        self,
        buffer_size: int = config.BUFFER_SIZE,
        acm_critic: bool = config.ACM_CRITIC,
        next_obs_diff = None,
        *args,
        **kwargs,
    ):
        f"""Off policy acm meta class

        Args:
            buffer_size (int, optional): Size of the replay buffer.
                Defautls to { config.BUFFER_SIZE }.
            acm_critic (bool): use actions of acm for critic.
                Defaults to { config.ACM_CRITIC }.
            next_obs_diff (list, optional): indices for which difference next_obs - obs should be computed 
                instead of absolute next_obs.
                Defaults to {None} 
            refill_buffer (bool, optional): if buffer should be refilled with new observations, when its full
                Defaults to {False}
        """
        super().__init__(*args, **kwargs)

        self.buffer_size = buffer_size
        self.acm_critic = acm_critic
        self.next_obs_diff = next_obs_diff

        #offset for the replay buffer
        REPLAY_BUFFER_OFFSET = 100

        self.replay_buffer = BufferAcMOffPolicy(
            self.buffer_size + REPLAY_BUFFER_OFFSET,
            self.ob_dim,
            self.actor_output_dim,
            acm_act_shape=self.ac_dim,
            acm_discrete=self.discrete,
            dtype=torch.float32,
            device=self.device,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            min_max_denormalize=self.min_max_denormalize,
            max_obs=self.max_obs,
            min_obs=self.min_obs,
            obs_norm=self.obs_norm,
        )

        new_hparams = {
            "hparams/buffer_size": self.buffer_size,
            "hparams/acm_critic": self.acm_critic,
        }
        self.hparams_acm.update(new_hparams)   

    def compute_next_obs_diff(self, obs: torch.tensor, next_obs: torch.tensor) -> torch.tensor:
        r = next_obs.clone()        
        r[:, self.next_obs_diff] = next_obs[:, self.next_obs_diff] - obs[:, self.next_obs_diff]
        return r


    def initial_act(self, obs) -> torch.Tensor:
        action = self.actor_ac_lim * torch.randn(1, self.actor_output_dim, device=self.device)
        if self.denormalize_actor_out:
            action = self.replay_buffer.denormalize(action, 
            self.acm_ob_idx)
        return action

    def collect_samples(self):
        """Collect samples into buffer for the pre-train stage
            In contrast do DDPG loop here we are adding next_obs instead of actions.
        """
        
        collected = 0
        while collected < self.acm_pre_train_samples:
            obs = self.env.reset()
            end = False
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            obs = obs.unsqueeze(0)

            prev_idx = self.replay_buffer.add_obs(obs)
            ep_len = 0

            while not end:
                acm_action = AcMTrainer.initial_act(self, obs)
                self.replay_buffer.add_acm_action(acm_action)
                prev_obs = obs
                obs, rew, done, _ = self.env.step(acm_action)
                ep_len += 1

                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)                
                obs = obs.unsqueeze(0)
                if self.next_obs_diff is not None:
                    obs = self.compute_next_obs_diff(prev_obs, obs)                

                end = True if ep_len == self.max_ep_len else done
                done = False if ep_len == self.max_ep_len else done

                next_idx = self.replay_buffer.add_obs(obs)
                self.replay_buffer.add_timestep(prev_idx, next_idx, self.cut_obs(obs), rew, done, end)
                prev_idx = next_idx
                collected += 1                


    def process_action(
        self, action: torch.Tensor, obs: torch.tensor, pre_train: bool = False
    ):
        """Pre-processing of action before it will go the env.

        Args:
            action (torch.Tensor): action from the policy.
            obs (torch.tensor): observations for this actions.

        Returns:
            np.array: processed action
        """
        with torch.no_grad():
            acm_observation = torch.cat([obs, action], axis=1)
            acm_action = self.acm.act(acm_observation)
            acm_action = acm_action.cpu().numpy()[0]
            self.replay_buffer.add_acm_action(acm_action)
        return acm_action

    def add_tensorboard_logs(self, buffer: MemoryAcM, done: bool):
        super().add_tensorboard_logs(buffer, done)
