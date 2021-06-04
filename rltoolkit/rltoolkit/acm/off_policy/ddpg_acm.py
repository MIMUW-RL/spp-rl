import numpy as np
import torch
from torch.nn import functional as F

from rltoolkit.acm.off_policy import AcMOffPolicy
from rltoolkit.algorithms import DDPG
from rltoolkit.algorithms.ddpg.models import Actor, Critic


class DDPG_AcM(AcMOffPolicy, DDPG):
    def __init__(
        self, unbiased_update: bool = False, custom_loss: float = 0.0, 
        custom_loss_target: float = 0.0, custom_loss_lr: float = 0.0001, 
        refill_buffer: bool = False,
        lagrangian_custom_loss: bool = False, separate_custom_loss: bool = False,
        cw_cl_targets: list = None, custom_loss_target_decay: int = None, 
        custom_loss_target_dfactor: float = None,
        *args, **kwargs,
    ):
        f"""DDPG with AcM class

        Args:
            unbiased_update (bool, optional): Use next_obs as action for update.
                Defaults to { False }.
            refill_buffer (bool, optional): if buffer should be refilled with new observations, when its full
                Defaults to {False} 

        """
        super().__init__(*args, **kwargs)
        self.unbiased_update = unbiased_update
        self.actor = Actor(
            self.ob_dim, ac_lim=self.actor_ac_lim, ac_dim=self.actor_output_dim
        )
        if not self.acm_critic:
            self.critic = Critic(self.ob_dim, ac_dim=self.actor_output_dim)

        self.custom_loss = custom_loss
        custom_loss_scaled = np.log(np.exp(custom_loss) - 1)
        self.custom_loss_param = torch.tensor(custom_loss_scaled) if not separate_custom_loss else torch.Tensor([custom_loss_scaled] * self.actor_output_dim)
        self.custom_loss_param.requires_grad = lagrangian_custom_loss
        self.custom_loss_target = custom_loss_target
        self.cw_cl_targets = cw_cl_targets
        if lagrangian_custom_loss and cw_cl_targets:
            self.custom_loss_target = cw_cl_targets
        self.lagrangian_custom_loss = lagrangian_custom_loss
        self.custom_loss_lr = custom_loss_lr
        self.separate_custom_loss = separate_custom_loss
        self.custom_loss_optimizer = self.opt([self.custom_loss_param], lr=custom_loss_lr)
        self.refill_buffer = refill_buffer
        self.custom_loss_target_decay = custom_loss_target_decay
        self.custom_loss_target_dfactor = custom_loss_target_dfactor

        if self.custom_loss:
            self.loss["ddpg"] = 0.0
            self.loss["dist"] = 0.0
            if lagrangian_custom_loss:
                if self.separate_custom_loss:
                    self.distances = []
                    for i in range(self.actor_output_dim):
                        self.loss[f"custom_loss_param/{i}"] = 0.0
                else:
                    self.loss["custom_loss_param"] = 0.0

        new_hparams = {
            "hparams/unbiased_update": self.unbiased_update,
            "hparams/custom_loss": self.custom_loss,
            "hparams/lagrangian_cl": self.lagrangian_custom_loss,
            "hparams/custom_loss_target_decay": self.custom_loss_target_decay,
            "hparams/custom_loss_target_dfactor": self.custom_loss_target_dfactor,
        }
        if self.lagrangian_custom_loss:
            if self.cw_cl_targets is None:
                new_hparams["hparams/cl_target"] = self.custom_loss_target
            new_hparams["hparams/cl_lr"] = self.custom_loss_lr

        self.hparams_acm.update(new_hparams)
        self.hparams.update(self.hparams_acm)

    def noise_action(self, obs, act_noise, deterministic=False):
        action, _ = self._actor.act(obs, deterministic)
        noise = act_noise * torch.randn(self.actor_output_dim, device=self.device)
        action += noise * self.actor_ac_lim
        action = np.clip(
            action.cpu(), -1.1 * self.actor_ac_lim.cpu(), 1.1 * self.actor_ac_lim.cpu()
        )
        action = action.to(self.device)
        if self.denormalize_actor_out:
            action = self.replay_buffer.denormalize(action, self.acm_ob_idx)
        return action

    def custom_loss_target_decay_condition(self):
        return(
            self.custom_loss_target_decay is not None
            and self.custom_loss_target_dfactor is not None
            and self.iterations > 0
            and self.stats_logger.frames % self.custom_loss_target_decay == 0
        )

    def acm_update_condition(self):
        return (
            self.iteration > 0
            and self.acm_epochs > 0
            and self.stats_logger.frames % self.acm_update_freq == 0
        )

    def make_unbiased_update(self):
        if self.update_condition():
            for _ in range(self.grad_steps):
                batch = self.replay_buffer.sample_batch(
                    self.update_batch_size, self.device
                )
                obs, next_obs, _, reward, done, acm_action = batch
                self.update(
                    obs=obs,
                    next_obs=next_obs,
                    action=next_obs,
                    reward=reward,
                    done=done,
                    acm_action=acm_action,
                )

    def make_update(self):
        if self.unbiased_update:
            self.make_unbiased_update()
        else:
            super().make_update()

        if self.custom_loss_target_decay_condition():            
            self.custom_loss_target *= self.custom_loss_target_dfactor
            print(f"CUSTOM LOSS TARTGET DECAY, CURRENT VALUE {self.custom_loss_target}")

        if self.acm_update_condition():
            if self.acm_update_batches:
                self.update_acm_batches(self.acm_update_batches)
            else:
                self.update_acm(self.acm_epochs)

    def collect_params_dict(self):
        params_dict = super().collect_params_dict()
        params_dict["acm"] = self.acm.state_dict()
        return params_dict

    def apply_params_dict(self, params_dict):
        super().apply_params_dict(params_dict)
        self.acm.load_state_dict(params_dict["acm"])

    def save_model(self, save_path=None):
        save_path = DDPG.save_model(self, save_path)
        torch.save(self.acm.state_dict(), save_path + "_acm_model.pt")

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
            next_action, _ = self.actor_targ(next_obs)
            next_action = self.replay_buffer.denormalize(next_action, self.acm_ob_idx)
            if self.acm_critic:
                acm_obs = torch.cat([next_obs, next_action], axis=1)
                next_action = self.acm(acm_obs)
            q_target = self.critic_targ(next_obs, next_action)

            qfunc_target = reward + self.gamma * (1 - done) * q_target

        return qfunc_target
    
    def add_custom_loss(self, loss, action, denorm_action, next_obs):
        if self.custom_loss:
            self.loss["ddpg"] = loss.item()
            if self.norm_closs:
                next_obs = self.replay_buffer.normalize(next_obs, force=True)
            else:
                action = denorm_action
            if not self.separate_custom_loss:
                loss_dist = F.mse_loss(action, self.cut_obs(next_obs))
                self.loss["dist"] = loss_dist.item()
                if self.lagrangian_custom_loss:
                    loss += F.softplus(self.custom_loss_param) * (loss_dist - self.custom_loss_target)
                else:
                    loss += self.custom_loss * loss_dist
                
                if self.custom_loss_target_decay is not None:
                    self.loss["custom_loss_target"] = self.custom_loss_target

            else:
                distances = torch.mean(F.mse_loss(action, self.cut_obs(next_obs), reduction='none'), dim=0)
                if self.cw_cl_targets is None:                    
                    loss += torch.sum(F.softplus(self.custom_loss_param) * (distances - self.custom_loss_target))
                else:                    
                    loss += torch.sum(F.softplus(self.custom_loss_param) * (distances - torch.Tensor(self.custom_loss_target)))

                self.loss["dist"] = distances.detach()
            
                if self.debug_mode:                                        
                    for j in range(distances.shape[0]):                        
                        self.loss[f"dist/cw/{j}"] = distances[j]

            return loss

    def compute_pi_loss(self, obs, next_obs):
        action, _ = self._actor(obs)
        denorm_action = self.replay_buffer.denormalize(action, self.acm_ob_idx)
        if self.acm_critic:
            acm_obs = torch.cat([obs, denorm_action], axis=1)
            critic_action = self.acm(acm_obs)
        else:
            critic_action = denorm_action
        loss = -self._critic(obs, critic_action).mean()

        return self.add_custom_loss(loss, action, denorm_action, next_obs)
    
    def update_custom_loss_param_loss(self):
        if not self.lagrangian_custom_loss:
            return
        dist_loss = self.loss["dist"]
        if self.cw_cl_targets is None:
            loss = -F.softplus(self.custom_loss_param) * (dist_loss - self.custom_loss_target)
        else:
            loss = -F.softplus(self.custom_loss_param) * (dist_loss - torch.Tensor(self.custom_loss_target))
        if self.separate_custom_loss:
            for i in range(len(loss)):
                self.loss[f"custom_loss_param/{i}"] = loss[i].item()
            self.loss["dist"] = torch.mean(self.loss["dist"]).item()
            loss = torch.sum(loss)
        else:
            self.loss["custom_loss_param"] = loss.item()
        self.custom_loss_optimizer.zero_grad()
        loss.backward()
        self.custom_loss_optimizer.step()


    def copy_offline_dataset(self, dataset, size):
        """copies the provided offlineRL dataset into the replay buffer.
            for the moment assumes D4RL dataset format (a dictionary) 
            and copies elements one-by-one
        """        
        i = 0
        traj = 0
        while i < size:
            traj += 1
            done  = torch.tensor(dataset['timeouts'][i] or dataset['terminals'][i])
            obs = torch.tensor(dataset['observations'][i])
            prev_idx = self.replay_buffer.add_obs(obs)
            i += 1
            ep_len = 0
            while(not done and i < size):

                nextobs = torch.tensor(dataset['observations'][i])            

                rew = torch.tensor( dataset['rewards'][i] )
                done = torch.tensor( dataset['timeouts'][i] or dataset['terminals'][i] )
                action = torch.tensor( dataset['actions'][i] )                
                end = torch.tensor( dataset['terminals'][i] )
                next_idx = self.replay_buffer.add_obs(nextobs)
                self.replay_buffer.add_timestep(
                    prev_idx, next_idx, nextobs, rew, done, end
                )
                self.replay_buffer.add_acm_action(action)                
                prev_idx = next_idx
                i += 1
                ep_len += 1            

        print(f"copied offline dataset with {i} samples, contains {traj} trajectories")
        #sets the internal variables according to the provided offline dataset
        self.acm_pre_train_samples = i
        self.buffer_size = i
        self.max_frames = i
        self.iterations = i / self.steps_per_epoch
        #updates std/dev/min/max parameters of the dataset
        self.update_obs_mean_std(self.replay_buffer)


    def collect_batch_and_train(self, steps_per_epoch: int, *args, **kwargs):
        """SPP variant of rollouts and collect samples if there is enough samples 
            in replay buffer use existing samples to perform actor/critic update
            otherwise generate new samples till steps_per_epoch number of steps
            will be added to the replay buffer

        Args:
            steps_per_epoch (int): number of samples to collect and train
            *args, **kwargs: arguments for make_update
        """
        collected = 0
        while collected < steps_per_epoch:
            

            # important part, 
            # when the replay buffer is filled stop generating new frames, just use the existing buffer
            # such that the number of used experience in learning is counted correctly
            if (self.stats_logger.frames >= self.buffer_size - self.acm_pre_train_samples) and not self.refill_buffer: 
                self.stats_logger.frames += 1
                collected += 1
                self.make_update(*args, **kwargs)
                continue
            

            self.stats_logger.rollouts += 1        
            obs = self.env.reset()
            # end - end of the episode from perspective of the simulation
            # done - end of the episode from perspective of the model
            end = False
            obs = self.process_obs(obs)
            prev_idx = self.replay_buffer.add_obs(obs)
            ep_len = 0

            while not end:
                
                obs = self.replay_buffer.normalize(obs)
                if (self.stats_logger.frames > self.acm_pre_train_samples) and (self.stats_logger.frames <= self.acm_pre_train_samples + self.random_frames):
                    action = self.initial_act(obs)
                else:
                    action = self.noise_action(obs, self.act_noise)
                action_proc = self.process_action(action, obs)
                prev_obs = obs

                obs, rew, done, _ = self.env.step(action_proc)
                ep_len += 1
                end = True if ep_len == self.max_ep_len else done
                done = False if ep_len == self.max_ep_len else done

                obs = self.process_obs(obs)                
                if self.next_obs_diff is not None:     
                    obs = self.compute_next_obs_diff(prev_obs, obs)
            
                next_idx = self.replay_buffer.add_obs(obs)
                
                
                self.replay_buffer.add_timestep(
                    prev_idx, next_idx, action, rew, done, end
                )
                prev_idx = next_idx



                self.stats_logger.frames += 1
                collected += 1                

                self.make_update(*args, **kwargs)

    def update(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        acm_action: torch.Tensor,
    ):
        """DDPG update step

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

        # Update Q-function by one step
        y_q = self._critic(obs, action)
        loss_q = F.mse_loss(y_q, y)

        self.loss["critic"] = loss_q.item()

        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()

        # Update policy by one step
        self._critic.eval()

        loss = self.compute_pi_loss(obs, next_obs)
        self.loss["actor"] = loss.item()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        #update temperature of Lagrangian optimization obj
        self.update_custom_loss_param_loss()

        # Update target networks

        self.update_target_nets()

        self._critic.train()

        for param in self.acm.parameters():
            param.requires_grad = True

    def add_tensorboard_logs(self, buffer, done):
        super().add_tensorboard_logs(buffer, done)
        if self.lagrangian_custom_loss:
            self.tensorboard_writer.log_custom_loss_param(
                self.iteration, self.custom_loss_param)

if __name__ == "__main__":
    #with torch.cuda.device(0):
    model = DDPG_AcM(
        # unbiased_update=True,
        # custom_loss=True,
        # acm_update_batches=50,
        # denormalize_actor_out=True,
        env_name="Pendulum-v0",
        buffer_size=50000,
        act_noise=0.05,
        iterations=100,
        gamma=0.99,
        steps_per_epoch=200,
        stats_freq=5,
        test_episodes=3,
        custom_loss=1,
        lagrangian_custom_loss=False,
        # tensorboard_dir="logs_ddpg",
        # tensorboard_comment="",
        acm_update_freq=200,
        acm_epochs=1,
        acm_pre_train_epochs=10,
        acm_pre_train_samples=10000,
        use_gpu=False,
        render=False,
    )
    model.pre_train()
    model.train()
