import copy
import tempfile
import os

import numpy as np
import pytest
import torch

from rltoolkit.algorithms.ppo.ppo import PPO
from rltoolkit.buffer import Memory


def test_cartpole():
    env_name = "CartPole-v0"
    iterations = 20
    stats_freq = 40
    return_done = 40
    model = PPO(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        tensorboard_comment="test",
        return_done=return_done,
        gamma=0.99,
        ppo_batch_size=200,
        max_ppo_epochs=10,
        actor_lr=1e-3,
        critic_lr=3e-4,
    )
    model.train()
    assert model.stats_logger.running_return > return_done, "Not solved by PPO"


CLIP_TEST_CASES = [
    (
        torch.tensor([-2.3, -5, -1.4, -1.5], requires_grad=True),
        torch.tensor([-2.3, -5, -1.4, -1.5], requires_grad=True),
        torch.tensor([1, 2.0, 3.0, 4.0]),
        -2.5,
    ),
    (
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([-1]),
        1,
    ),
    (
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([-2.0], requires_grad=True),
        torch.tensor([-1.0]),
        0.8,
    ),
    (
        torch.tensor([-2.0], requires_grad=True),
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([1.0]),
        -1.2,
    ),
    (
        torch.tensor([-1.0], requires_grad=True),
        torch.tensor([-2.0], requires_grad=True),
        torch.tensor([1.0]),
        -0.3679,
    ),
]


@pytest.mark.parametrize(
    "action_logprobs, new_logprobs, advantages, expected_result", CLIP_TEST_CASES
)
def test_clip_loss(action_logprobs, new_logprobs, advantages, expected_result):
    ppo = PPO(epsilon=0.2)
    result = ppo._clip_loss(action_logprobs, new_logprobs, advantages)

    assert expected_result == pytest.approx(result.item(), 0.0001)
    assert result.requires_grad


@pytest.fixture()
def memory():
    buffer = Memory()
    observations = np.ones((13, 2)) * 5
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


def test_calculate_gae(memory):
    expected_q_value = torch.tensor(
        [5.0, 6.0, 7.0, 3.0, 9.0, 10.0, 6.0, 12.0, 13.0, 14.0, 6.0, 2.0]
    )
    expected_gae_adv = np.array(
        [-6.2969, -5.1875, -4.75, -7, -1.25, -1, -4, 3.1562, 4.6250, 6.5, -6, -8]
    )

    def critic_func(obs, *args):
        if len(obs.shape) == 2:
            return obs[:, 0] * 2
        else:
            return obs[0] * 2

    ppo = PPO(gamma=0.5, gae_lambda=0.5)
    ppo._critic = critic_func
    memory.obs_mean = torch.zeros(2)
    memory.obs_std = torch.ones(2)
    q_value = ppo.calculate_q_val(memory)
    assert torch.equal(expected_q_value.float(), q_value.float())

    result = ppo.calculate_gae(memory, q_value).numpy()
    np.testing.assert_almost_equal(expected_gae_adv, result, decimal=4)


def test_save_and_load():
    env_name = "Pendulum-v0"
    iterations = 3
    stats_freq = 3
    return_done = -1000
    tmpdir = tempfile.TemporaryDirectory()
    save_dir = os.path.join(tmpdir.name, "test_model" + ".pkl")
    model = PPO(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        return_done=return_done,
    )
    model.train()
    model.obs_mean = -100
    model.obs_std = 100
    model.save(path=save_dir)

    loaded_model = PPO(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        return_done=return_done,
    )
    loaded_model.load(save_dir)
    assert model.obs_mean == -100 and model.obs_std == 100

    tmpdir.cleanup()
