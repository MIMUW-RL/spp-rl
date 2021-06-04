import logging

import torch
from torch.utils.data import DataLoader

from rltoolkit import config, utils
from rltoolkit.algorithms.a2c.a2c import A2C
from rltoolkit.algorithms.ppo.advantage_dataset import AdvantageDataset
from rltoolkit.buffer import Memory

logger = logging.getLogger(__name__)


class PPO(A2C):
    def __init__(
        self,
        epsilon: float = config.PPO_EPSILON,
        gae_lambda: float = config.GAE_LAMBDA,
        kl_div_threshold: float = config.PPO_MAX_KL_DIV,
        max_ppo_epochs: int = config.PPO_MAX_EPOCHS,
        ppo_batch_size: int = config.PPO_BATCH_SIZE,
        entropy_coef: float = config.PPO_ENTROPY,
        *args,
        **kwargs,
    ):
        f"""
        Proximal Policy Optimization implementation.

        Args:
            epsilon (float, optional): Clipping PPO parameter. Ratio between a new
                action probability and old one will be clipped to the range
                <1 - epsilon, 1 + epsilon>  Defaults to { config.PPO_EPSILON }.
            gae_lambda (float, optional): Generalized Advantage Estimation lambda
                parameter. Defaults to { config.GAE_LAMBDA }.
            kl_div_threshold (float, optional): Maximal KL divergence between old and
                new policy. If this treshold is acquired agent's update is stopped in
                this iteration. Defaults to { config.PPO_MAX_KL_DIV }.
            max_ppo_epochs (int, optional): Maximal number of SGD steps without
                exceeding KL divergence threshold.
                Defaults to { config.PPO_MAX_EPOCHS }.
            ppo_batch_size (int, optional): PPO batch size in SGD.
                Defaults to { config.PPO_BATCH_SIZE }.
            entropy_coef (float, optional): weight of entropy in the actor loss
                function. Defaults to { config.PPO_ENTROPY }.
            actor_lr (float, optional): Learning rate of the actor.
                Defaults to { config.A_LR }.
            critic_lr (float, optional): Learning rate of the critic.
                Defaults to { config.C_LR }.
            critic_num_target_updates (int, optional): Number of target q-value updates
                per iteration. Defaults to { config.NUM_TARGET_UPDATES }.
            num_critic_updates_per_target (int, optional): Number of gradient steps of
                critic for one target. Defaults to { config.NUM_CRITIC_UPDATES }.
            normalize_adv (bool, optional): Normalize advantages for actor.
                Defaults to { config.NORMALIZE_ADV }.
            obs_norm_alpha (float, optional): If set describes how much variance and
                mean should be updated each iteration. obs_norm_alpha == 1 means no
                update and obs_norm_alpha == 0 means replace old mean and std.
                Defaults to { config.NORM_ALPHA }
            env_name (str, optional): Name of the gym environment.
                Defaults to { config.ENV_NAME }.
            gamma (float, optional): Discount factor. Defaults to { config.GAMMA }.
            stats_freq (int, optional): Frequency of logging the progress.
                Defaults to { config.STATS_FREQ }.
            steps_per_epoch (int, optional): Number of frames used for one algorithm step
                (could be higher because batch collection stops when rollout ends).
                Defaults to { config.STEPS_PER_EPOCH }.
            iterations (int, optional): Number of algorithms iterations.
                Defaults to { config.ITERATIONS }.
            max_frames (int, optional): Limit of frames for training. Defaults to
                { None }.
            return_done (Union[int, None], optional): target return, which will stop
                training if reached. Defaults to { config.RETURN_DONE }.
            log_dir (str, optional): Path for basic logs which includes final model.
                Defaults to { config.LOG_DIR }.
            use_gpu (bool, optional): Use CUDA. Defaults to { config.USE_GPU }.
            tensorboard_dir (Union[str, None], optional): Path to tensorboard logs.
                Defaults to { config.TENSORBOARD_DIR }.
            tensorboard_comment (str, optional): Comment for tensorboard files.
                Defaults to { config.TENSORBOARD_COMMENT }.
            verbose (int, optional): Verbose level. Defaults to { config.VERBOSE }.
            render (bool, optional): Render rollouts to tensorboard.
                Defaults to { config.RENDER }.
        """
        super().__init__(*args, **kwargs)
        self.ppo_epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.kl_div_threshold = kl_div_threshold
        self.max_ppo_epochs = max_ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.entropy_coef = entropy_coef
        self.kl_div_updates_counter = 0
        new_hparams = {
            "hparams/ppo_epsilon": self.ppo_epsilon,
            "hparams/gae_lambda": self.gae_lambda,
            "hparams/kl_div_threshold": self.kl_div_threshold,
            "hparams/max_ppo_epochs": self.max_ppo_epochs,
            "hparams/ppo_batch_size": self.ppo_batch_size,
        }
        self.hparams.update(new_hparams)

    def calculate_advantage(self, buffer: Memory) -> torch.tensor:
        """
        Estimate advantage using GAE.

        Args:
            buffer (Memory): memory with samples

        Returns:
            torch.tensor: tensor containing GAE advantages for observations from buffer
        """
        with torch.no_grad():
            q_val = self.calculate_q_val(buffer)
            advantage = self.calculate_gae(buffer, q_val)

        return advantage

    def calculate_gae(self, buffer: Memory, q_val: torch.tensor) -> torch.tensor:
        """
        Calculate advanatage using Generalized Advantage Estimation (GAE).

        Args:
            buffer (Memory): memory with samples
            q_val (torch.tensor): tensor containing Q values for observations from
                the buffer

        Returns:
            torch.tensor: tensor containing GAE advantages for observations from buffer
        """
        obs = buffer.norm_obs
        next_obs = buffer.norm_next_obs
        ends = torch.tensor(buffer.end, dtype=torch.float32, device=self.device)
        dones = torch.tensor(buffer.done, dtype=torch.float32, device=self.device)

        state_value = self.critic(obs)

        deltas = q_val - state_value.squeeze()
        advantage = torch.empty(size=deltas.size())

        discount = self.gae_lambda * self.gamma

        for i, delta in enumerate(reversed(deltas)):
            idx = -i - 1
            if dones[idx]:
                gae = 0
            elif ends[idx]:
                gae = self.critic(next_obs[idx]).squeeze().item()
            gae = gae * discount + delta
            advantage[idx] = gae
        advantage = advantage.to(self.device)
        return advantage

    def update_actor(self, advantages: torch.Tensor, buffer: Memory):
        """One iteration of actor update

        Args:
            advantages (torch.Tensor): advantages for observations from buffer
            buffer (Memory): buffer with samples
        """
        advantage_dataset = AdvantageDataset(advantages, buffer, self.normalize_adv)
        dataloader = DataLoader(
            advantage_dataset, batch_size=self.ppo_batch_size, shuffle=True
        )
        kl_div = 0.0
        self.loss["actor"] = 0
        self.loss["entropy"] = 0
        self.loss["sum"] = 0

        for i in range(self.max_ppo_epochs):
            if kl_div >= self.kl_div_threshold:
                break

            for advantages, action_logprobs, actions, norm_obs in dataloader:

                new_dist = self.actor.get_actions_dist(norm_obs)
                new_logprobs = new_dist.log_prob(torch.squeeze(actions.detach()))

                entropy = new_dist.entropy().mean()

                actor_loss = self._clip_loss(action_logprobs, new_logprobs, advantages)
                loss = actor_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.loss["actor"] += actor_loss.item()
                self.loss["entropy"] += entropy.item()
                self.loss["sum"] += loss.item()

            kl_div = utils.kl_divergence(action_logprobs.cpu(), new_logprobs.cpu())

        logger.debug(f"PPO update finished after {i} epochs with KL = {kl_div}")
        self.kl_div_updates_counter += i + 1

    def _clip_loss(
        self,
        action_logprobs: torch.tensor,
        new_logprobs: torch.tensor,
        advantages: torch.tensor,
    ):
        ratio = torch.exp(new_logprobs - action_logprobs.detach())
        clipped_ratio = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
        # minus sign is needed to do gradient ascent
        actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
        return actor_loss

    def add_tensorboard_logs(self, *args, **kwargs):
        super().add_tensorboard_logs(*args, **kwargs)
        if self.debug_mode:
            self.log_ppo_epochs()

    def log_ppo_epochs(self):
        if self.iteration == 0:
            mean_updates = self.kl_div_updates_counter
        elif self.iteration % self.stats_freq == 0:
            mean_updates = self.kl_div_updates_counter / self.stats_freq
        else:
            denominator = self.iteration % self.stats_freq
            mean_updates = self.kl_div_updates_counter / denominator

        self.tensorboard_writer.log_kl_div_updates(
            self.iteration,
            self.stats_logger.frames,
            self.stats_logger.rollouts,
            mean_updates,
        )
        self.kl_div_updates_counter = 0


if __name__ == "__main__":
    model = PPO(
        env_name="Hopper-v2",
        iterations=1000,
        max_frames=1e6,
        gamma=0.99,
        actor_lr=3e-4,
        critic_lr=1e-3,
        steps_per_epoch=2000,
        ppo_batch_size=256,
        kl_div_threshold=0.15,
        stats_freq=5,
        max_ppo_epochs=50,
        tensorboard_dir="logs",
    )
    model.train()
