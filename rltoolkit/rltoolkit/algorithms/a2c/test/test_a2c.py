import copy

import numpy as np
import pytest
import torch

from rltoolkit.algorithms.a2c.a2c import A2C
from rltoolkit.buffer import Memory


@pytest.fixture()
def memory():
    buffer = Memory()
    observations = np.ones((13, 2))
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


def test_calculate_q_value(memory):
    expected_result = torch.tensor(
        [5.0, 6.0, 7.0, 3.0, 9.0, 10.0, 6.0, 12.0, 13.0, 14.0, 6.0, 2.0]
    )

    def critic_func(*args):
        return torch.tensor([10])

    a2c = A2C(gamma=0.5)
    a2c._critic = critic_func
    result = a2c.calculate_q_val(memory)

    assert torch.equal(expected_result, result)


def test_get_obs_mean_std():
    model = A2C(obs_norm_alpha=None)
    mean, std = model._get_initial_obs_mean_std(None)
    assert mean is None
    assert std is None

    model = A2C(obs_norm_alpha=0.9)
    model.ob_dim = 2
    mean, std = model._get_initial_obs_mean_std(0.9)
    np.testing.assert_array_equal(mean.numpy(), np.zeros(2))
    np.testing.assert_array_equal(std.numpy(), np.ones(2))
