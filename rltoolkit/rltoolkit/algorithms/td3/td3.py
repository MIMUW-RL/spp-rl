import copy
import logging
from itertools import chain

import numpy as np
import torch
from torch.nn import functional as F

from rltoolkit.algorithms.ddpg import DDPG
from rltoolkit.algorithms.ddpg.models import Critic

logger = logging.getLogger(__name__)


class TD3(DDPG):
    def __init__(
        self, *args, pi_update_freq: int = 2, noise_clip: float = 0.5, policy_noise: float = 0.2, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._critic_1 = None
        self.critic_1_optimizer = None
        self.critic_1_targ = None
        self._critic_2 = None
        self.critic_2_optimizer = None
        self.critic_2_targ = None

        self.pi_update_freq = pi_update_freq
        self.noise_clip = noise_clip

        self.critic_1 = Critic(self.ob_dim, self.ac_dim)
        self.critic_2 = Critic(self.ob_dim, self.ac_dim)

        self.loss = {"actor": 0.0, "critic_1": 0.0, "critic_2": 0.0}
        new_hparams = {
            "hparams/pi_update_freq": self.pi_update_freq,
        }
        self.hparams.update(new_hparams)

    @property
    def critic_1(self):
        return self._critic_1

    @critic_1.setter
    def critic_1(self, model: torch.nn.Module):
        self._critic_1, self.critic_1_optimizer = self.set_model(model, self.critic_lr)
        self.critic_1_targ = copy.deepcopy(self._critic_1)

    @property
    def critic_2(self):
        return self._critic_2

    @critic_2.setter
    def critic_2(self, model: torch.nn.Module):
        self._critic_2, self.critic_2_optimizer = self.set_model(model, self.critic_lr)
        self.critic_2_targ = copy.deepcopy(self._critic_2)

    def compute_qfunc_targ(
        self, reward: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor
    ):
        """Compute targets for Q-functions

        Args:
            reward (torch.Tensor): batch of rewards
            next_obs (torch.Tensor): batch of next observations
            done (torch.Tensor): batch of done

        Returns:
            torch.Tensor: Q-function targets for the batch
        """
        with torch.no_grad():
            action, _ = self.actor_targ(next_obs)
            noise = np.clip(
                self.policy_noise * torch.randn(self.ac_dim, device=self.device),
                -self.noise_clip,
                self.noise_clip,
            )
            action += noise
            action = np.clip(action.cpu(), -self.ac_lim.cpu(), self.ac_lim.cpu()).to(
                self.device
            )
            q1_target = self.critic_1_targ(next_obs, action)
            q2_target = self.critic_2_targ(next_obs, action)
            q_target = torch.min(q1_target, q2_target)

            qfunc_target = reward + self.gamma * (1 - done) * q_target

        return qfunc_target

    def compute_pi_loss(self, obs):
        """Loss for the policy

        Args:
            obs (torch.Tensor): batch of observations

        Returns:
            torch.Tensor: policy loss
        """
        action, _ = self._actor(obs)
        loss = -self._critic_1(obs, action).mean()
        return loss

    def update_target_nets(self):
        """Update target networks with Polyak averaging
        """
        with torch.no_grad():
            # Polyak averaging:
            critics_params = chain(
                self._critic_1.parameters(),
                self._critic_2.parameters(),
                self._actor.parameters(),
            )
            targets_params = chain(
                self.critic_1_targ.parameters(),
                self.critic_2_targ.parameters(),
                self.actor_targ.parameters(),
            )
            for q_params, targ_params in zip(critics_params, targets_params):
                targ_params.data.mul_(1 - self.tau)
                targ_params.data.add_((self.tau) * q_params.data)

    def update(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        """TD3 update step

        Args:
            obs (torch.Tensor): observations tensor
            next_obs (torch.Tensor): next observations tensor
            action (torch.Tensor): actions tensor
            reward (torch.Tensor): rewards tensor
            done (torch.Tensor): dones tensor
        """
        y = self.compute_qfunc_targ(reward, next_obs, done)

        # Update Q-function by one step
        y_q1 = self._critic_1(obs, action)
        loss_q1 = F.mse_loss(y_q1, y)
        y_q2 = self._critic_2(obs, action)
        loss_q2 = F.mse_loss(y_q2, y)

        self.loss["critic_1"] = loss_q1.item()
        self.loss["critic_2"] = loss_q2.item()

        self.critic_1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic_2_optimizer.step()

        # Update policy if needed
        if self.stats_logger.frames % (self.update_freq * self.pi_update_freq) == 0:
            self._critic_1.eval()
            self._critic_2.eval()

            loss = self.compute_pi_loss(obs)
            self.loss["actor"] = loss.item()

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            self._critic_1.train()
            self._critic_2.train()

        # Update target networks
        self.update_target_nets()

    def collect_params_dict(self):
        params_dict = {}
        params_dict["actor"] = self.actor.state_dict()
        params_dict["critic_1"] = self.critic_1.state_dict()
        params_dict["critic_2"] = self.critic_2.state_dict()
        params_dict["obs_mean"] = self.replay_buffer.obs_mean
        params_dict["obs_std"] = self.replay_buffer.obs_std
        params_dict["min_obs"] = self.replay_buffer.min_obs
        params_dict["max_obs"] = self.replay_buffer.max_obs
        return params_dict

    def apply_params_dict(self, params_dict):
        self.actor.load_state_dict(params_dict["actor"])
        self.critic_1.load_state_dict(params_dict["critic_1"])
        self.critic_2.load_state_dict(params_dict["critic_2"])
        self.obs_mean = params_dict["obs_mean"]
        self.obs_std = params_dict["obs_std"]
        self.min_obs = params_dict["min_obs"]
        self.max_obs = params_dict["max_obs"]
        self.replay_buffer.obs_mean = self.obs_mean
        self.replay_buffer.obs_std = self.obs_std
        self.replay_buffer.min_obs = self.min_obs
        self.replay_buffer.max_obs = self.max_obs

    def save_model(self, save_path=None) -> str:
        if self.filename is None and save_path is None:
            raise AttributeError
        elif save_path is None:
            save_path = str(self.log_path)

        torch.save(self._actor.state_dict(), save_path + "_actor_model.pt")
        torch.save(self._critic_1.state_dict(), save_path + "_critic_1_model.pt")
        torch.save(self._critic_2.state_dict(), save_path + "_critic_2_model.pt")
        return save_path


if __name__ == "__main__":
    with torch.cuda.device(0):
        torch.set_num_threads(1)
        model = TD3(
            env_name="HalfCheetah-v2",
            buffer_size=int(1e6),
            iterations=1000,
            gamma=0.99,
            steps_per_epoch=1000,
            stats_freq=5,
            random_frames=1000,
            test_episodes=3,
            use_gpu=False,
            obs_norm=False,
            pi_update_freq=2,
            tensorboard_dir="td3_logs",
            tensorboard_comment="",
            log_dir="td3_logs",
        )
        model.train()
        # model.save("tmp_norb.pkl")
