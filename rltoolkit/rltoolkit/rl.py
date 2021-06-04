import logging
import pickle as pkl
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import gym
import torch

from rltoolkit import config, utils
from rltoolkit.buffer import Memory
from rltoolkit.stats_logger import StatsLogger
from rltoolkit.tensorboard_logger import TensorboardWriter

logger = logging.getLogger(__name__)

class MetaLearner:
    def __init__(
        self,
        env_name: str,
        use_gpu: bool,
        debug_mode: bool = config.DEBUG_MODE,
        tensorboard_dir: Union[str, None] = config.TENSORBOARD_DIR,
        tensorboard_comment: str = config.TENSORBOARD_COMMENT,
        min_max_denormalize: bool = config.MIN_MAX_DENORMALIZE,
    ):
        f"""Class with parameters common for RL and other interactions with environment

        Args:
            env_name (str): Name of the gym environment.
            use_gpu (bool): Use CUDA.
            debug_mode (bool, optional): Log additional info.
                Defaults to { config.DEBUG_MODE }
            tensorboard_dir (Union[str, None], optional): Path to tensorboard logs.
                Defaults to { config.TENSORBOARD_DIR }.
            tensorboard_comment (str, optional): Comment for tensorboard files.
                Defaults to { config.TENSORBOARD_COMMENT }.
        """
        self.env_name = env_name
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.envnames_ant = ['AntBlock', 'AntBlockMaze', 'AntFall', 'AntMaze', 'AntPush']
        if self.env_name in self.envnames_ant:
            from environments.create_maze_env import create_maze_env
            from environments.env_with_goal import EnvWithGoal
            
            self.env = EnvWithGoal(create_maze_env(self.env_name), self.env_name )            
        else:
            self.env = gym.make(self.env_name)
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.ob_dim = self.env.observation_space.shape[0]
        if self.discrete:
            self.ac_dim = self.env.action_space.n
            self.ac_lim = None
        else:
            self.ac_dim = self.env.action_space.shape[0]
            self.ac_lim = torch.tensor(self.env.action_space.high, device=self.device)

        self.obs_mean = torch.zeros(self.ob_dim, device=self.device)
        self.obs_std = torch.ones(self.ob_dim, device=self.device)
        self.max_obs = None
        self.min_obs = None

        self.iteration = 0  # used in tensorboard
        self.last_success_rate = None

        self.opt = torch.optim.Adam
        self.loss = {}

        self.debug_mode = debug_mode

        self.tensorboard_writer = None
        self.tensorboard_comment = (
            "_" + tensorboard_comment if tensorboard_comment else ""
        )
        self.tensorboard_dir = tensorboard_dir
        self.min_max_denormalize = min_max_denormalize
        self.normalized_buffer = False

    def run_tensorboard_if_needed(self):
        if self.tensorboard_writer is None and (self.tensorboard_dir is not None):
            self.tensorboard_writer = TensorboardWriter(
                env_name=self.env_name,
                log_dir=self.tensorboard_dir,
                filename=self.filename,
                render=self.render,
            )

    def log_obs_mean_std_tensorboard(self):
        """
        Log mean and std of observations in the tensorboard.
        """
        self.run_tensorboard_if_needed()
        self.tensorboard_writer.log_obs_mean_std(
            self.iteration, self.obs_mean, self.obs_std
        )

    def update_obs_mean_std(self, buffer: Memory) -> Memory:
        """
        Update running average of mean and stds based on the buffer.
        Also update max and min of the observations.

        Args:
            buffer (Memory)

        Returns:
            Memory
        """
        buffer.update_obs_mean_std()
        self.obs_mean = buffer.obs_mean
        self.obs_std = buffer.obs_std
        self.max_obs = buffer.max_obs
        self.min_obs = buffer.min_obs

        if self.debug_mode and self.tensorboard_dir is not None:
            self.log_obs_mean_std_tensorboard()
        return buffer


class RL(MetaLearner):
    def __init__(
        self,
        env_name: str = config.ENV_NAME,
        gamma: float = config.GAMMA,
        stats_freq: int = config.STATS_FREQ,
        test_episodes: int = config.TEST_EPISODES,
        steps_per_epoch: int = config.STEPS_PER_EPOCH,
        iterations: int = config.ITERATIONS,
        max_frames: int = None,
        return_done: Union[int, None] = config.RETURN_DONE,
        log_dir: str = config.LOG_DIR,
        use_gpu: bool = config.USE_GPU,
        verbose: int = config.VERBOSE,
        render: bool = config.RENDER,
        max_episode_steps: int = None,
        *args,
        **kwargs,
    ):
        f"""Basic parent class for reinforcement learning algorithms.

        Args:
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
            verbose (int, optional): Verbose level. Defaults to { config.VERBOSE }.
            render (bool, optional): Render rollouts to tensorboard.
                Defaults to { config.RENDER }.
            debug_mode (bool, optional): Log additional info.
                Defaults to { config.DEBUG_MODE }
            tensorboard_dir (Union[str, None], optional): Path to tensorboard logs.
                Defaults to { config.TENSORBOARD_DIR }.
            tensorboard_comment (str, optional): Comment for tensorboard files.
                Defaults to { config.TENSORBOARD_COMMENT }.
        """
        super().__init__(env_name, use_gpu, *args, **kwargs)
        assert iterations > 0, f"Iteration has to be positive not {iterations}"
        if max_frames is not None:
            assert (
                max_frames <= iterations * steps_per_epoch
            ), "max_frames should be smaller or equal than iterations * steps_per_epoch"

        self.max_frames = max_frames
        self.gamma = gamma
        self.stats_freq = stats_freq
        self.test_episodes = test_episodes
        self.steps_per_epoch = steps_per_epoch
        self.iterations = iterations
        self.return_done = return_done
        if log_dir is not None:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = log_dir
        self.verbose = verbose
        self.render = render

        
        self.max_ep_len = max_episode_steps if max_episode_steps is not None else self.env._max_episode_steps

        self.start_time = utils.get_time()

        self.hparams = {
            "hparams/gamma": self.gamma,
            "hparams/steps_per_epoch": self.steps_per_epoch,
            "hparams/type": utils.get_pretty_type_name(self),
        }
        self.shortnames = config.SHORTNAMES
        self.stats_logger = StatsLogger()

    def train(self, iterations=None):
        f""" Train RL model

        Args:
            iterations ([type], optional): Number of additional training iterations.
            If None performs number of iterations defined in self.iterations.
            Otherwise increase global counter by this value to run additional steps.
            Defaults to { None }.
        """
        self.run_tensorboard_if_needed()
        if iterations:
            self.iterations += iterations

        while self.iteration < self.iterations:            
            buffer, time_diff = self.perform_iteration()

            self.stats_logger.time_list.append(time_diff)                        
            running_return = self.stats_logger.calc_running_return(buffer)            
            if self.return_done is not None and running_return >= self.return_done:
                break
            
            if self.iteration % self.stats_freq == 0:
                self.logs_after_iteration(buffer)

            if self.log_dir is not None:
                self.stats_logger.dump_stats(self.log_path)

            self.iteration += 1  # used also for logs
            if (
                self.max_frames is not None
                and self.max_frames <= self.stats_logger.frames
            ):
                logger.info(f"Reached max_frames at {self.iteration} iteration")  # INFO
                break

        self.logs_after_iteration(buffer, done=True)

        if self.log_dir is not None:
            self.save()

    def test(self, episodes=None):
        f"""Test policy

        Args:
            episodes (int): Number of episodes. Defaults to { None }.

        Returns:
            float: mean episode reward
        """
        mean_reward = None
        return mean_reward

    @utils.measure_time
    def perform_iteration(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def check_path(self, path):
        if self.filename is None and path is None:
            raise AttributeError
        elif path is None:
            path = str(self.log_path) + ".pkl"
        return path

    def collect_params_dict(self):
        params_dict = {}
        params_dict["actor"] = self.actor.state_dict()
        params_dict["critic"] = self.critic.state_dict()
        params_dict["obs_mean"] = self.obs_mean
        params_dict["obs_std"] = self.obs_std
        params_dict["min_obs"] = self.min_obs
        params_dict["max_obs"] = self.max_obs
        return params_dict

    def apply_params_dict(self, params_dict):
        self.actor.load_state_dict(params_dict["actor"])
        self.critic.load_state_dict(params_dict["critic"])
        self.obs_mean = params_dict["obs_mean"]
        self.obs_std = params_dict["obs_std"]
        self.min_obs = params_dict["min_obs"]
        self.max_obs = params_dict["max_obs"]

    def save(self, path: str = None):
        """Save RL object

        Args:
            path (str): Path to save
        """
        path = self.check_path(path)
        with open(path, "wb") as f:
            params_dict = self.collect_params_dict()
            pkl.dump(params_dict, f)

    def load(self, path: str):
        """Load RL object

        Args:
            path (str): Path to saved RL object
        """
        path = self.check_path(path)
        with open(path, "rb") as f:
            params_dict = pkl.load(f)
            self.apply_params_dict(params_dict)

    @property
    def log_iteration(self):
        return self.iteration // self.stats_freq

    @property
    def filename(self):
        suffix = self.get_tensorboard_hparams_suffix()
        suffix += self.tensorboard_comment
        filename = self.start_time + suffix
        return filename

    @property
    def log_path(self):
        log_path = Path(self.log_dir)
        log_path = log_path / self.filename
        return log_path

    def logs_after_iteration(self, buffer: Memory, done: bool = False):
        f"""Logs writer

        Args:
            buffer (Memory): Buffer used for tensorboard
            done (bool, optional): Finalize tensorboard logging due to last iteration.
            Defaults to { False }.
        """
        if self.test_episodes is not None:
            self.stats_logger.test_return = self.test()
            if self.env_name in self.envnames_ant:
                self.stats_logger.success_rate = self.last_success_rate

        running_return = self.stats_logger.running_return
        if self.verbose:
            if done:
                self.stats_logger.task_done(self.iteration)
            else:
                self.stats_logger.log_stats(self.iteration)

        self.stats_logger.stats.append([self.iteration, running_return])
        self.stats_logger.reset_time_list()

        if self.tensorboard_writer is not None:
            self.add_tensorboard_logs(buffer, done)

    def add_tensorboard_logs(self, buffer: Memory, done: bool ):
        self.tensorboard_writer.log_running_return(
            self.iteration,
            self.stats_logger.frames,
            self.stats_logger.rollouts,
            self.stats_logger.running_return,
        )
        if self.test_episodes:
            self.tensorboard_writer.log_test_return(
                self.iteration,
                self.stats_logger.frames,
                self.stats_logger.rollouts,
                self.stats_logger.test_return,
            )

        if (self.log_iteration % 5) == 0 or done:
            _, rendering_time = self.tensorboard_writer.record_episode(
                self, self.iteration, done
            )
        self.tensorboard_writer.log_returns(self.iteration, buffer)
        self.tensorboard_writer.log_actions(
            self.iteration, buffer, denormalize=self.normalized_buffer
        )
        self.tensorboard_writer.log_observations(self.iteration, buffer)
        self.tensorboard_writer.log_loss(self.iteration, self.loss)
        if self.stats_logger.success_rate is not None:
            self.tensorboard_writer.log_success_rate( self.stats_logger.frames, self.stats_logger.success_rate )

    def get_tensorboard_hparams_suffix(self):
        suffix = self.env_name
        
        for key, val in self.hparams.items():
            if key in self.shortnames.keys():
                key, default_value = self.shortnames[key]
                if val == default_value:
                    continue  # skip default values
            else:
                key = key.split("/")[1]

            if isinstance(val, float):
                val = f"{val:.2}"
            else:
                val = str(val)
            suffix += f"-{key}{val}"

        return suffix

    def _get_initial_obs_mean_std(
        self, obs_norm: Any
    ) -> Tuple[Optional[torch.tensor], Optional[torch.tensor]]:
        """
        Check if observations are normalized and if so return initial mean and std,
        None otherwise.

        Returns:
            Tuple[Optional[torch.tensor], Optional[torch.tensor]]: obs mean and std
        """
        if obs_norm:
            obs_mean = torch.zeros(self.ob_dim, device=self.device)
            obs_std = torch.ones(self.ob_dim, device=self.device)
        else:
            obs_mean = None
            obs_std = None
        return obs_mean, obs_std
