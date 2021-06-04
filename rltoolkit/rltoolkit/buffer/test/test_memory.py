import copy

import numpy as np
import pytest
import torch

from rltoolkit.buffer.memory import Memory


@pytest.fixture()
def memory():
    buffer = Memory()
    observations = torch.tensor([[i, 10 * i] for i in range(1, 14)]).float()
    actions = np.ones((12, 2))
    rewards = np.array([i for i in range(10)] + [1, 2])
    dones = np.zeros(12)
    actions = (actions.T * rewards).T
    dones[3] = 1
    dones[6] = 1
    dones[11] = 1
    ends = copy.deepcopy(dones)
    ends[9] = 1

    rollouts = 0
    i = 0
    while rollouts < 4:
        rollouts += 1
        obs = observations[i]
        end = False
        prev_idx = buffer.add_obs(torch.tensor(obs).unsqueeze(dim=0))
        while not end:
            action = actions[i]
            obs, rew, done, end = observations[i + 1], rewards[i], dones[i], ends[i]
            next_idx = buffer.add_obs(torch.tensor(obs).unsqueeze(dim=0))
            buffer.add_timestep(prev_idx, next_idx, action, actions, rew, done, end)
            prev_idx = next_idx
            i += 1
        buffer.end_rollout()
    buffer.update_obs_mean_std()
    return buffer


def test_obs(memory):
    expected_result = torch.tensor([[i, 10 * i] for i in range(1, 13)]).float()
    result = memory.obs
    assert torch.equal(expected_result, result)


def test_next_obs(memory):
    expected_result = torch.tensor([[i, 10 * i] for i in range(2, 14)]).float()
    result = memory.next_obs
    assert torch.equal(expected_result, result)


def test_obs_norm(memory):
    mean = torch.tensor([7.1875, 71.8750])
    std = torch.tensor([3.6737, 36.7367]) + 1e-8
    expected_result = torch.tensor([[i, 10 * i] for i in range(1, 13)]).float()
    expected_result = (expected_result - memory.obs_mean) / (memory.obs_std + 1e-8)
    result = memory.norm_obs

    assert torch.equal(mean, memory.obs_mean)
    assert std[0].item() == pytest.approx(memory.obs_std[0].item(), 4)
    assert std[1].item() == pytest.approx(memory.obs_std[1].item(), 4)
    assert torch.equal(expected_result, result)


def test_next_obs_norm(memory):
    mean = torch.tensor([7.1875, 71.8750])
    std = torch.tensor([3.6737, 36.7367]) + 1e-8
    expected_result = torch.tensor([[i, 10 * i] for i in range(2, 14)]).float()
    expected_result = (expected_result - memory.obs_mean) / (memory.obs_std + 1e-8)
    result = memory.norm_next_obs

    assert torch.equal(mean, memory.obs_mean)
    assert std[0].item() == pytest.approx(memory.obs_std[0].item(), 4)
    assert std[1].item() == pytest.approx(memory.obs_std[1].item(), 4)
    assert torch.equal(expected_result, result)


def test_normalize(memory):
    memory.obs_std = torch.tensor([2.0, 20.0])
    memory.obs_mean = torch.tensor([2.5, 25.0])
    example = torch.tensor([[i, 10 * i] for i in range(6)]).float()
    result = memory.normalize(example)
    expected_result = torch.tensor([[(i - 2.5) / 2, (i - 2.5) / 2] for i in range(6)])
    assert torch.equal(expected_result, result)

    example[0, 0] = 1000
    result = memory.normalize(example)
    expected_result[0, 0] = 10
    assert torch.equal(expected_result, result), "Outlier fail"
