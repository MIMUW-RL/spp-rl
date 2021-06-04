import logging
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from rltoolkit import config
from rltoolkit.basic_model import AcM
from rltoolkit.buffer import MemoryAcM, ReplayBufferAcM
from rltoolkit.rl import MetaLearner

logger = logging.getLogger(__name__)


class AcMTrainer(MetaLearner):
    def __init__(
        self,
        env_name: str = config.ENV_NAME,
        acm_epochs: int = config.ACM_EPOCHS,
        acm_batch_size: int = config.ACM_BATCH_SIZE,
        acm_update_freq: int = config.ACM_UPDATE_FREQ,
        acm_ob_idx: list = config.ACM_OB_IDX,
        acm_lr: float = config.ACM_LR,
        acm_pre_train_samples: int = config.ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs: int = config.ACM_PRE_TRAIN_N_EPOCHS,
        acm_scheduler_step: int = config.ACM_SCHEDULER_STEP,
        acm_scheduler_gamma: int = config.ACM_SCHEDULER_GAMMA,
        acm_fn= AcM,
        use_gpu: bool = config.USE_GPU,
        acm_update_batches: Optional[int] = config.ACM_UPDATE_BATCHES,
        denormalize_actor_out: bool = config.DENORMALIZE_ACTOR_OUT,
        acm_keep_pretrain: bool = config.ACM_KEEP_PRE_TRAIN,
        obs_norm: bool = config.OBS_NORM,
        norm_closs: bool = config.NORM_CLOSS,
        *args,
        **kwargs,
    ):
        f"""Acm implementation

        Args:
            env_name (str, optional): Name of the gym environment.
                Defaults to { config.ENV_NAME }.
            acm_epochs (int, optional): Number of epochs per Action Model update.
                Defaults to { config.ACM_EPOCHS }.
            acm_batch_size (int, optional): Batch size for Action Model update.
                Defaults to { config.ACM_BATCH_SIZE }.
            acm_update_freq (int, optional): Frequency of Action Model update
                (per iterations). Defaults to { config.ACM_UPDATE_FREQ }.
            acm_ob_idx (list, optional): Observation indexes used for Action Model.
                Defaults to { config.ACM_OB_IDX }.
            acm_lr (float, optional): Learning rate of Action Model.
                Defaults to { config.ACM_LR }.
            acm_pre_train_samples (int, optional): Number of samples for pre-training.
                Defaults to { config.ACM_PRE_TRAIN_SAMPLES }.
            acm_pre_train_epochs (int, optional): Number of epochs for pre-training.
                Defaults to { config.ACM_PRE_TRAIN_N_EPOCHS }.
            acm_scheduler_step (int, optional): Scheduler update step for the optimizer
                learning rate. Defaults to { config.ACM_SCHEDULER_STEP }.
            acm_scheduler_gamma (int, optional): Scheduler gamma factor for the
                optimizer learning rate. Defaults to { config.ACM_SCHEDULER_GAMMA }.            
            acm_fn: Function that creates control model.
            use_gpu (bool, optional): Flag activating CUDA.
                Defaults to { config.USE_GPU }.
            acm_update_batches (int, optional): If None use whole replay buffer and
                update in epochs, otherwise do only acm_update_batches batches update.
                Defaults to { config.ACM_UPDATE_BATCHES }.
            denormalize_actor_out (bool, optional): If False actions from Actor will be
                clipped, otherwise they also will be denormalized based on mean and std
                from replay buffer. Defaults to { config.DENORMALIZE_ACTOR_OUT }.
            acm_keep_pretrain (bool, optional): If True samples from pre-train will be
                keeped in replay buffer for futher updates (in off-policy also will be
                used to update actor). If False pre-train will drop buffer after
                finished. Defaults to { config.ACM_KEEP_PRE_TRAIN }.
            obs_norm (bool): If true actor input is normalized.
                Defaults to { config.OBS_NORM }.
            norm_closs (bool): If true custom loss is normalized.
                Defaults to { config.NORM_CLOSS }.
            debug_mode (bool, optional): Log additional info.
                Defaults to { config.DEBUG_MODE }.
            tensorboard_dir (Union[str, None], optional): Path to tensorboard logs.
                Defaults to { config.TENSORBOARD_DIR }.
            tensorboard_comment (str, optional): Comment for tensorboard files.
                Defaults to { config.TENSORBOARD_COMMENT }.
        """
        super().__init__(env_name=env_name, use_gpu=use_gpu, *args, **kwargs)
        self._acm = None

        self.acm_epochs = acm_epochs
        self.acm_batch_size = acm_batch_size
        self.acm_update_freq = acm_update_freq

        self.acm_ob_idx = acm_ob_idx
        if self.acm_ob_idx is None:  # use the whole observation vector
            self.acm_ob_idx = list(range(self.ob_dim))
        else:
            assert max(acm_ob_idx) < self.ob_dim, "acm_ob_idx out of range"

        self.denormalize_actor_out = denormalize_actor_out

        action_lims = self.env.observation_space.high
        if self.min_max_denormalize:
            action_lims = 1.0  # do not consider outliers for stability
        elif self.denormalize_actor_out or any(action_lims == float("inf")):
            action_lims = config.MAX_ABS_OBS_VALUE

        self.actor_ac_lim = torch.tensor(action_lims, device=self.device)
        self.actor_output_dim = len(self.acm_ob_idx)

        self.acm_lr = acm_lr
        self.acm_pre_train_samples = acm_pre_train_samples
        self.acm_pre_train_epochs = acm_pre_train_epochs
        self.acm_update_batches = acm_update_batches
        self.acm_keep_pretrain = acm_keep_pretrain

        if self.discrete:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.MSELoss()

        self.obs_norm = obs_norm
        self.norm_closs = norm_closs

        overhead = 0.1  # 10% overhead in buffer for terminal observations
        self.buffer_size = int(self.acm_pre_train_samples * (1 + overhead))
        self.replay_buffer = ReplayBufferAcM(
            self.buffer_size,
            self.ob_dim,
            self.ac_dim,
            self.discrete,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            min_max_denormalize=self.min_max_denormalize,
            max_obs=self.max_obs,
            min_obs=self.min_obs,
            obs_norm=self.obs_norm,
        )
        
        self.in_dim = self.ob_dim + len(self.acm_ob_idx)

        self.acm_lr = acm_lr
        self.acm_scheduler_step = acm_scheduler_step
        self.acm_scheduler_gamma = acm_scheduler_gamma

        self.acm = acm_fn(self.in_dim, self.ac_dim, self.ac_lim, self.discrete)

        self.loss["acm"] = 0.0

        self.hparams_acm = {
            "hparams/acm_epochs": self.acm_epochs,
            "hparams/acm_batch_size": self.acm_batch_size,
            "hparams/acm_update_freq": self.acm_update_freq,
            "hparams/acm_lr": self.acm_lr,
            "hparams/acm_pre_train_samples": self.acm_pre_train_samples,
            "hparams/acm_pre_train_epochs": self.acm_pre_train_epochs,
            "hparams/denormalize_actor_out": self.denormalize_actor_out,
            "hparams/acm_update_batches": self.acm_update_batches,
            "hparams/acm_keep_pretrain": self.acm_keep_pretrain,
            "hparams/min_max_denormalize": self.min_max_denormalize,
            "hparams/norm_closs": self.norm_closs,
        }

    @property
    def acm(self):
        return self._acm

    @acm.setter
    def acm(self, model: torch.nn.Module):
        self._acm = model
        self._acm.to(device=self.device)
        self.acm_optimizer = self.opt(self._acm.parameters(), lr=self.acm_lr)
        self.acm_scheduler = torch.optim.lr_scheduler.StepLR(
            self.acm_optimizer, self.acm_scheduler_step, self.acm_scheduler_gamma
        )

    def initial_act(self, obs: torch.Tensor):
        """Actor for samples collection

        Args:
            obs (torch.Tensor): observation from the environment.

        Raises:
            action (np.array): action used for environment step.
        """
        action = self.env.action_space.sample()
        return action

    def collect_samples(self):
        """Collect samples into buffer for the pre-train stage
        """
        self.replay_buffer = self.collect_initial_batch(
            self.replay_buffer, self.acm_pre_train_samples
        )

    def collect_initial_batch(
        self, buffer: ReplayBufferAcM, samples_no: int
    ) -> ReplayBufferAcM:
        raise NotImplementedError

    def pre_train(self):
        """Pre-train acm model"""

        self.collect_samples()        
        self.update_acm(epochs=self.acm_pre_train_epochs, pretrain=True)        
        self.update_obs_mean_std(self.replay_buffer)
        if not self.acm_keep_pretrain:
            self.replay_buffer.reset_idx()

    def batch_update(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        if not self.discrete and y.dim() < 2:
            y = y.reshape(len(y), -1)

        y_pred = self.acm(x)
        loss = self.loss_fn(y_pred, y)
        self.acm_optimizer.zero_grad()
        loss.backward()

        self.acm_optimizer.step()
        return loss.item()
    
    def cut_obs(self, obs):
        return obs[:, self.acm_ob_idx]

    def acm_cat(self, obs, next_obs):
        acm_obs = torch.cat(
            [obs, next_obs[:, self.acm_ob_idx]], axis=1
        )
        return acm_obs

    def update_acm(self, epochs: int, pretrain: bool = False):
        """Single Action Model update

        Args:
            epochs (int): number of epochs
            pretrain (bool): True in pretrain for tb error plots. Defaults to { False }
        """
        obs = torch.tensor(self.replay_buffer.obs, dtype=torch.float32)
        next_obs = torch.tensor(self.replay_buffer.next_obs, dtype=torch.float32)
        if self.discrete:
            actions_acm = torch.tensor(self.replay_buffer.actions_acm, dtype=torch.long)
        else:
            actions_acm = torch.tensor(
                self.replay_buffer.actions_acm, dtype=torch.float32
            )

        acm_obs = self.acm_cat(obs, next_obs)
        new_data = TensorDataset(acm_obs, actions_acm)

        loader = DataLoader(new_data, batch_size=self.acm_batch_size, shuffle=True)
        for e in range(epochs):
            epoch_loss = 0
            for i, (x, y) in enumerate(loader):
                loss = self.batch_update(x, y)
                epoch_loss += loss

            epoch_loss /= i + 1
            if self.debug_mode:                
                logger.debug("Epoch %d, loss = %.4f" % (e, epoch_loss))

            self.acm_scheduler.step()
        self.loss["acm"] = epoch_loss


    def save_model(self, filename):
        torch.save(self.acm.state_dict(), filename)

    def add_tensorboard_logs(self, buffer: MemoryAcM, done: bool):
        super().add_tensorboard_logs(buffer, done)
        if self.debug_mode:
            self.tensorboard_writer.log_acm_action_histogram(self.iteration, buffer)
            self.tensorboard_writer.log_action_mean_std(
                self.iteration, buffer, False, self.acm_ob_idx
            )

    def update_acm_batches(self, n_batches: int, epochs = 1):
        update_loss = 0
        for e in range(epochs):
            for _ in range(n_batches):
                obs, next_obs, actions_acm = self.replay_buffer.sample_acm_batch(
                    self.acm_batch_size
                )
                acm_obs = self.acm_cat(obs, next_obs)
                loss = self.batch_update(acm_obs, actions_acm)
                update_loss += loss
            update_loss /= n_batches

            logger.debug("Update loss(epoch %d), loss = %.4f" % (e, update_loss))                    

        self.loss["acm"] = update_loss        

if __name__ == "__main__":
    env_name = "Pendulum-v0"
    acm_pre_train_samples = 1000
    acm_pre_train_epochs = 10
    model = AcMTrainer(
        env_name=env_name,
        acm_pre_train_samples=acm_pre_train_samples,
        acm_pre_train_epochs=acm_pre_train_epochs,
    )
    model.pre_train()
