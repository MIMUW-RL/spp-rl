import logging
from collections import defaultdict
from os import path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from rltoolkit.tensorboard_logger import TensorboardWriter

logger = logging.getLogger(__name__)

class EvalsWrapper:
    def __init__(
        self,
        Algo: type,
        evals: int,
        tensorboard_dir: str,
        log_all: bool = False,
        torch_threads: int = 1,  # scaling, more is worse :D
        *args,
        **kwargs,
    ):
        torch.set_num_threads(torch_threads)
        self.Algo = Algo
        self.evals = evals
        self.args = args
        if log_all:
            kwargs.update({"tensorboard_dir": path.join(tensorboard_dir, "tb_runs")})
        self.kwargs = kwargs
        self.tensorboard_dir = path.join(tensorboard_dir, "tb_hparams")
        self.hparams = {}
        self.metrics = defaultdict(list)
        self.filename = None

    def perform_evaluations(self):
        for i in range(self.evals):
            algo = self.Algo(**self.kwargs)
            if i == 0:
                logger.info("Started %s", algo.filename)                
            algo.train()
            self.metrics["frames"].append(algo.stats_logger.frames)
            self.metrics["returns"].append(algo.stats_logger.running_return)
            self.metrics["iterations"].append(algo.iteration)
            if "test_episodes" in self.kwargs:
                self.metrics["test_return"].append(algo.stats_logger.test_return)
        self.filename = algo.filename
        self.hparams = algo.hparams
        logger.info("Ended %s", algo.filename)

    def update_tensorboard(self):
        metrics = {}
        for key, val in self.metrics.items():
            key = f"metrics/{key}"
            arr = np.array(val)
            mean = arr.mean()
            std = arr.std()
            metrics[key + "_mean"] = mean
            metrics[key + "_std"] = std

        writer = TensorboardWriter(
            env_name=None,
            log_dir=self.tensorboard_dir,
            filename=self.filename,
            render=False,
        )
        hparams_numeric, hparams_other = self.split_hparams_to_numeric_and_other()
        metrics.update(hparams_numeric)
        writer.log_hyperparameters(hparams_other, metrics)

    def split_hparams_to_numeric_and_other(self) -> Tuple[dict, dict]:
        numeric = {k: v for k, v in self.hparams.items() if isinstance(v, (float, int))}
        other = {
            k: v for k, v in self.hparams.items() if not isinstance(v, (float, int))
        }
        return numeric, other


class EvalsWrapperACM(EvalsWrapper):
    def __init__(self, acm_model: Optional[nn.Module] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acm_model = acm_model

    def perform_evaluations(self):
        for i in range(self.evals):
            algo = self.Algo(**self.kwargs)
            if i == 0:
                logger.info("Started %s", algo.filename)
            if self.acm_model:
                algo.acm = self.acm_model

            algo.pre_train()
            algo.train()
            self.metrics["frames"].append(algo.stats_logger.frames)
            self.metrics["returns"].append(algo.stats_logger.running_return)
            self.metrics["iterations"].append(algo.iteration)
            if "test_episodes" in self.kwargs:
                self.metrics["test_return"].append(algo.stats_logger.test_return)
        self.filename = algo.filename
        self.hparams = algo.hparams
        logger.info("Ended %s", algo.filename)


class EvalsWrapperACMOffline(EvalsWrapper):
    def __init__(self, acm_model: Optional[nn.Module] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acm_model = acm_model        

    def perform_evaluations(self, dataset, size):
        for i in range(self.evals):

            algo = self.Algo(**self.kwargs)
            if i == 0:                
                logger.info("Started %s", algo.filename)
            if self.acm_model:
                algo.acm = self.acm_model

            #copy the provided dataset into the replay buffer
            algo.copy_offline_dataset(dataset, size)
            #then pretrain AcM            
            #algo.update_acm_batches(n_batches = algo.acm_update_batches, epochs = algo.acm_pre_train_epochs)
            algo.train()
            
            self.metrics["frames"].append(algo.stats_logger.frames)
            self.metrics["returns"].append(algo.stats_logger.running_return)
            self.metrics["iterations"].append(algo.iteration)
            if "test_episodes" in self.kwargs:
                self.metrics["test_return"].append(algo.stats_logger.test_return)
        self.filename = algo.filename
        self.hparams = algo.hparams
        logger.info("Ended %s", algo.filename)


if __name__ == "__main__":
    from rltoolkit import PPO

    ALGO = PPO
    EVALS = 1
    ITERATIONS = 200
    RETURN_DONE = 190
    GAMMA = 0.99
    A_LR = 1e-3
    C_LR = 1e-3
    STEPS_PER_EPOCH = 500
    TENSORBOARD_DIR = "tensorboard"
    LOG_DIR = "basic_logs"
    N_CORES = 4
    kwargs = {
        "Algo": ALGO,
        "evals": EVALS,
        "iterations": ITERATIONS,
        "tensorboard_dir": TENSORBOARD_DIR,
        "return_done": RETURN_DONE,
        "log_dir": LOG_DIR,
        "verbose": 1,
        "render": False,
        "test_episodes": 5,
    }
    evals = EvalsWrapper(**kwargs)
    evals.perform_evaluations()
    evals.update_tensorboard()
