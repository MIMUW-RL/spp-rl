import numpy as np
import torch
from torch.nn import functional as F

from rltoolkit.acm.off_policy.ddpg_acm import DDPG_AcM
from rltoolkit.algorithms import TD3
from rltoolkit.algorithms.ddpg.models import Actor, Critic


class TD3_AcM(DDPG_AcM, TD3):
    def __init__(self, *args, **kwargs):
        """TD3 with AcM implementation
        """
        super().__init__(*args, **kwargs)
        if not self.acm_critic:
            self.critic_1 = Critic(self.ob_dim, ac_dim=self.actor_output_dim)
            self.critic_2 = Critic(self.ob_dim, ac_dim=self.actor_output_dim)

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
                self.policy_noise * torch.randn(self.actor_output_dim, device=self.device),
                -self.noise_clip,
                self.noise_clip,
            )
            action += noise
            action = np.clip(
                action.cpu(), -self.actor_ac_lim.cpu(), self.actor_ac_lim.cpu()
            ).to(self.device)
            action = self.replay_buffer.denormalize(action, self.acm_ob_idx)
            if self.acm_critic:
                acm_obs = torch.cat([next_obs, action], axis=1)
                action = self.acm(acm_obs)
            q1_target = self.critic_1_targ(next_obs, action)
            q2_target = self.critic_2_targ(next_obs, action)
            q_target = torch.min(q1_target, q2_target)

            qfunc_target = reward + self.gamma * (1 - done) * q_target

        return qfunc_target

    def compute_pi_loss(self, obs, next_obs):
        """Loss for the policy

        Args:
            obs (torch.Tensor): batch of observations

        Returns:
            torch.Tensor: policy loss
        """
        action, _ = self._actor(obs)
        denorm_action = self.replay_buffer.denormalize(action, self.acm_ob_idx)
        if self.acm_critic:
            acm_obs = torch.cat([obs, denorm_action], axis=1)
            critic_action = self.acm(acm_obs)
        else:
            critic_action = denorm_action
        loss = -self._critic_1(obs, critic_action).mean()

        return self.add_custom_loss(loss, action, denorm_action, next_obs)

    def update(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        acm_action: torch.Tensor,
    ):
        """TD3 update step

        Args:
            obs (torch.Tensor): observations tensor
            next_obs (torch.Tensor): next observations tensor
            action (torch.Tensor): actions tensor
            reward (torch.Tensor): rewards tensor
            done (torch.Tensor): dones tensor
        """
        for param in self.acm.parameters():
            param.requires_grad = False

        if self.acm_critic:
            action = acm_action

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

            loss = self.compute_pi_loss(obs, next_obs)
            self.loss["actor"] = loss.item()

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            #update temperature of Lagrangian optimization obj
            self.update_custom_loss_param_loss()

            self._critic_1.train()
            self._critic_2.train()

        # Update target networks
        self.update_target_nets()

        for param in self.acm.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    #with torch.cuda.device(0):
    model = TD3_AcM(
        # unbiased_update=True,
        custom_loss=True,
        # acm_update_batches=50,
        # denormalize_actor_out=True,
        env_name="HalfCheetah-v2",
        buffer_size=50000,
        act_noise=0.05,
        iterations=100,
        gamma=0.99,
        steps_per_epoch=200,
        stats_freq=5,
        test_episodes=3,
        # tensorboard_dir="logs_ddpg",
        # tensorboard_comment="",
        acm_update_freq=200,
        acm_epochs=1,
        acm_pre_train_epochs=10,
        acm_pre_train_samples=10000,
        use_gpu=False,
        render=False,
        acm_critic=True,
    )
    model.pre_train()
    model.train()
