import torch
from torch.nn import functional as F

from rltoolkit.acm.off_policy.ddpg_acm import DDPG_AcM
from rltoolkit.algorithms import SAC
from rltoolkit.algorithms.sac.models import SAC_Actor, SAC_Critic


class SAC_AcM(DDPG_AcM, SAC):
    def __init__(self, act_noise: float = 0, *args, **kwargs):
        """SAC with AcM class
        """
        self.act_noise = act_noise
        super().__init__(*args, **kwargs)
        self.actor = SAC_Actor(
            self.ob_dim,
            ac_lim=self.actor_ac_lim,
            ac_dim=self.actor_output_dim,
            discrete=self.discrete,
        )
        if not self.acm_critic:
            self.critic_1 = SAC_Critic(
                self.ob_dim, ac_dim=self.actor_output_dim, discrete=self.discrete
            )
            self.critic_2 = SAC_Critic(
                self.ob_dim, ac_dim=self.actor_output_dim, discrete=self.discrete
            )
        self.hparams_acm["hparams/act_noise"] = self.act_noise

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
            sampled_next_action, sampled_next_logprob = self._actor(next_obs)
            sampled_next_action = self.replay_buffer.denormalize(sampled_next_action)
            if self.acm_critic:
                acm_obs = torch.cat([next_obs, sampled_next_action], axis=1)
                sampled_next_action = self.acm(acm_obs)

            q1_target = self.critic_1_targ(next_obs, sampled_next_action)
            q2_target = self.critic_2_targ(next_obs, sampled_next_action)
            q_target = torch.min(q1_target, q2_target)

            qfunc_target = reward + self.gamma * (1 - done) * (
                q_target - self.alpha * sampled_next_logprob
            )

        return qfunc_target

    def compute_pi_loss(
        self,
        obs: torch.Tensor,
        sampled_action: torch.Tensor,
        sampled_logprob: torch.Tensor,
        next_obs: torch.Tensor,
    ):
        denorm_action = self.replay_buffer.denormalize(sampled_action)
        if self.acm_critic:
            acm_obs = torch.cat([obs, denorm_action], axis=1)
            critic_action = self.acm(acm_obs)
        else:
            critic_action = denorm_action
        q1 = self._critic_1(obs, critic_action)
        q2 = self._critic_2(obs, critic_action)
        q = torch.min(q1, q2)

        loss = (self.alpha * sampled_logprob - q).mean()        
        return self.add_custom_loss(loss, sampled_action, denorm_action, next_obs)

    def update(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        acm_action: torch.Tensor,
    ):
        """Soft Actor-Critic update:

        Args:
            obs (torch.Tensor): observations tensor
            next_obs (torch.Tensor): next observations tensor
            action (torch.Tensor): actions tensor
            reward (torch.Tensor): rewards tensor
            done (torch.Tensor): dones tensor
            acm_action (torch.Tensor): tensor of acm actions
        """
        for param in self.acm.parameters():
            param.requires_grad = False

        if self.acm_critic:
            action = acm_action

        y = self.compute_qfunc_targ(reward, next_obs, done)

        # Update Q-functions by one step
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

        # Update policy by one step
        self._critic_1.eval()
        self._critic_2.eval()

        sampled_action, sampled_logprob = self._actor(obs)

        # if self.stats_logger.frames % (self.update_freq * self.pi_update_freq) == 0:
        loss = self.compute_pi_loss(obs, sampled_action, sampled_logprob, next_obs)
        self.loss["actor"] = loss.item()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_q()

        self._critic_1.train()
        self._critic_2.train()

        # Update temperature
        alpha_loss = self.compute_alpha_loss(sampled_logprob)

        #update temperature of Lagrangian optimization obj
        self.update_custom_loss_param_loss()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        for param in self.acm.parameters():
            param.requires_grad = True

    def save_model(self, save_path=None):
        save_path = SAC.save_model(self, save_path)
        torch.save(self.acm.state_dict(), save_path + "_acm_model.pt")


if __name__ == "__main__":
    #with torch.cuda.device(0):
    model = SAC_AcM(
        # unbiased_update=False,
        # custom_loss=1,
        # acm_update_batches=50,
        # normalize_actor_ac=False,
        env_name="HalfCheetah-v2",
        buffer_size=int(1e6),
        iterations=50,
        gamma=0.99,
        steps_per_epoch=1000,
        stats_freq=5,
        test_episodes=3,
        tensorboard_dir="logs_sac",
        tensorboard_comment="",
        update_freq=50,
        grad_steps=50,
        acm_update_freq=200,
        acm_epochs=1,
        custom_loss=1,
        acm_pre_train_epochs=10,
        acm_pre_train_samples=10000,
        use_gpu=False,
        render=False,
    )
    model.pre_train()
    model.train()
