import numpy as np
import pytest
import torch

from rltoolkit import utils


@pytest.fixture
def obs_mean_std():
    obs = torch.tensor(np.arange(20).reshape(4, 5).T).float()
    mean = obs.mean(axis=0)
    std = obs.std(axis=0)
    return obs, mean, std


def test_kl_divergence():
    log_p = torch.tensor([-0.73, -0.72, -0.45], dtype=torch.float64)
    log_q = torch.tensor([-0.57, -0.84, -0.13], dtype=torch.float64)

    expected_result = -0.12
    result = utils.kl_divergence(log_p, log_q)
    assert expected_result == pytest.approx(result, 0.0001)


def test_standardize_and_clip(obs_mean_std):
    obs, mean, std = obs_mean_std
    result = utils.standardize_and_clip(obs, mean, std)
    assert result.mean().item() + 1 == pytest.approx(1.0, 0.0001)
    assert result.std(0).mean().item() == pytest.approx(1.0, 0.0001)


def test_standardize_and_clip_with_clip(obs_mean_std):
    obs, mean, std = obs_mean_std
    obs[0, 0] = 100.0
    result = utils.standardize_and_clip(obs, mean, std, 4.0)
    assert result[0, 0] == 4.0


def test_revert_standardization(obs_mean_std):
    obs, mean, std = obs_mean_std
    obs_stand = utils.standardize_and_clip(obs, mean, std)
    result = utils.revert_standardization(obs_stand, mean, std)
    assert torch.equal(obs, result)
