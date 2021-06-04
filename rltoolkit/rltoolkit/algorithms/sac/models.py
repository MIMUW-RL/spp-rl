import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class SAC_Actor(nn.Module):
    def __init__(self, obs_dim: int, ac_lim: float, ac_dim: int, discrete: bool = True):
        super().__init__()

        self.ac_dim = ac_dim
        self.ac_lim = ac_lim
        self.obs_dim = obs_dim
        self.discrete = discrete
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_prob = nn.Linear(256, ac_dim)
        if not self.discrete:
            self.fc_scale = nn.Linear(256, ac_dim)
            self.log_scale_min = -20
            self.log_scale_max = 2

    def forward(self, x, deterministic=False):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        if self.discrete:
            # action_prob = torch.softmax(self.fc_prob(x), dim=1)
            action_prob = F.gumbel_softmax(self.fc_prob(x), dim=1)
            log_scale = None
            dist = Categorical(action_prob)
            if deterministic:
                action = torch.argmax(action_prob, dim=1)
            else:
                action = dist.sample()
            action_logprobs = dist.log_prob(action)
        else:
            action_mean = self.fc_prob(x)
            log_scale = self.fc_scale(x)
            log_scale = torch.clamp(log_scale, self.log_scale_min, self.log_scale_max)
            dist = Normal(action_mean, torch.exp(log_scale))
            if deterministic:
                action = action_mean
            else:
                action = dist.rsample()  # opposed to sample, rsample keeps grad
            action_logprobs = dist.log_prob(action).sum(axis=-1)

            # Correction for tanh squashing:
            correction = 2 * (np.log(2) - action - F.softplus(-2 * action)).sum(axis=1)
            action_logprobs -= correction
            action = torch.tanh(action)
            action = action * self.ac_lim

        return action, action_logprobs

    def act(self, obs, deterministic=False):
        f"""Returns actions without gradient

        Args:
            obs (torch.Tensor): Observations from env
            deterministic (bool, optional): If deterministic is True policy will perform
                greedy actions. Defaults to { False }.

        Returns:
            torch.Tensor: action
        """
        with torch.no_grad():
            action, action_logprobs = self.forward(obs, deterministic)
            return action, action_logprobs


class SAC_Critic(nn.Module):
    def __init__(self, obs_dim: int, ac_dim: int, discrete: bool):
        super().__init__()
        self.discrete = discrete
        self.ac_dim = ac_dim
        self.fc1 = nn.Linear(obs_dim + ac_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, ac):
        if self.discrete:
            ac = ac.unsqueeze(-1)
            one_hot = torch.zeros(ac.shape[0], self.ac_dim)
            one_hot.scatter_(1, ac, 1)
            ac = one_hot
        x = torch.cat((obs, ac), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.squeeze(x, -1)
