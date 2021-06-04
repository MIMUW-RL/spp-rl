import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from rltoolkit import config


def measure_time(func: Callable) -> Callable:
    def decorated(*args, **kwargs):
        time = datetime.datetime.now()
        result = func(*args, **kwargs)
        time_after = datetime.datetime.now()
        time_diff = time_after - time
        return result, time_diff.total_seconds()

    return decorated


def get_pretty_type_name(item: Any) -> str:
    t = str(type(item)).split("'")[1].split(".")[-1]
    return t


def get_log_dir(log_dir: str) -> Path:
    """
    Get directory name for new RL experiment.

    Arguments:
        log_dir {str} -- absolute or relative path to parent directory.

    Returns:
        Path -- new experiment tensorboard event path in the following form:
            "/path/to/log/dir/ + time"
    """
    log_dir = Path(log_dir)
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S.%f")[:-3]
    log_dir = log_dir / current_time
    return log_dir


def get_time() -> str:
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S.%f")[:-3]
    return current_time


def kl_divergence(log_p: torch.tensor, log_q: torch.tensor) -> torch.tensor:
    """
    Calculate KL divergence approximation of two distributions p and q.

    Args:
        p (torch.tensor): log probabilites from distribution 1
        q (torch.tensor): log probabilites from distribution 2

    Returns:
        torch.tensor: KL divergence >= 0
    """
    return (log_p - log_q).mean().item()


def standardize_and_clip(
    obs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    max_abs_value: float = config.MAX_ABS_OBS_VALUE,
) -> np.array:
    assert obs.shape[-1] == mean.shape[0]
    assert obs.shape[-1] == std.shape[0]
    obs_stand = (obs - mean) / (std + 1e-8)
    clipped_obs_stand = torch.clamp(obs_stand, -max_abs_value, max_abs_value)

    return clipped_obs_stand


def revert_standardization(
    obs_stand: np.array, mean: np.array, std: np.array
) -> np.array:
    assert obs_stand.shape[-1] == mean.shape[0]
    assert obs_stand.shape[-1] == std.shape[0]
    obs = obs_stand * std + mean

    return obs
