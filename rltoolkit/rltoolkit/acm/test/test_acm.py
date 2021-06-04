import gym
import numpy as np
import torch

from rltoolkit.acm.acm import AcMTrainer


def test_acm_discrete():
    env_name = "CartPole-v0"
    acm_pre_train_samples = 200
    acm_pre_train_epochs = 10
    model = AcMTrainer(
        env_name=env_name,
        acm_pre_train_samples=acm_pre_train_samples,
        acm_pre_train_epochs=acm_pre_train_epochs,
    )
    model.pre_train()


def test_acm_continuous():
    env_name = "Pendulum-v0"
    acm_pre_train_samples = 1000
    acm_pre_train_epochs = 10
    model = AcMTrainer(
        env_name=env_name,
        acm_pre_train_samples=acm_pre_train_samples,
        acm_pre_train_epochs=acm_pre_train_epochs,
    )
    model.pre_train()


def test_acm_accuracy():
    env_name = "CartPole-v0"
    evaluations = 100
    env = gym.make(env_name)
    acm_trainer = AcMTrainer(
        env_name=env_name, acm_pre_train_epochs=5, acm_pre_train_samples=1e3
    )
    acm_trainer.pre_train()

    acc = 0
    acm_actions = []
    i = 0
    while i < evaluations:
        done = False

        obs_1 = env.reset()
        while not done:
            act = env.action_space.sample()
            obs_2, _, done, _ = env.step(act)
            x = torch.tensor(np.concatenate([obs_1, obs_2]), dtype=torch.float32)[
                None, :
            ]
            acm_act = acm_trainer.acm.act(x).item()
            acm_actions.append(acm_act)

            acc += acm_act == act
            i += 1
            if i >= evaluations:
                acc /= evaluations
                break
            obs_1 = obs_2
    assert acc > 0.95
