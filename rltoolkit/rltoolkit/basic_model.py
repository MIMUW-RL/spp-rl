import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal


class Actor(nn.Module):
    def __init__(self, obs_dim: int, ac_lim: float, ac_dim: int, discrete: bool = True):
        super().__init__()

        self.ac_dim = ac_dim
        self.ac_lim = ac_lim
        self.obs_dim = obs_dim
        self.discrete = discrete
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, ac_dim)
        if not self.discrete:
            self.log_scale = nn.Parameter(
                -1.34 * torch.ones(self.ac_dim), requires_grad=True
            )

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        if self.discrete:
            x = torch.softmax(self.fc3(x), dim=1)
        else:
            x = torch.tanh(self.fc3(x))
        return x

    def act(self, obs, deterministic=False):
        if self.discrete:
            action_prob = self.forward(obs)
            dist = Categorical(action_prob)
            if deterministic:
                action = torch.argmax(action_prob, dim=1)
            else:
                action = dist.sample()
        else:
            action_mean = self.forward(obs)
            action_mean = action_mean * self.ac_lim
            normal = Normal(action_mean, torch.exp(self.log_scale))
            dist = Independent(normal, 1)
            if deterministic:
                action = action_mean.detach()
            else:
                action = dist.sample()
        action_logprobs = dist.log_prob(torch.squeeze(action))

        return action, action_logprobs

    def get_actions_dist(self, obs):
        if self.discrete:
            action_prob = self.forward(obs)
            dist = Categorical(action_prob)
        else:
            action_mean = self.forward(obs)
            action_mean = action_mean * self.ac_lim
            normal = Normal(action_mean, torch.exp(self.log_scale))
            dist = Independent(normal, 1)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorAcM(nn.Module):
    def __init__(self, obs_dim: int, ac_lim: float, ac_dim: int):
        super().__init__()

        self.ac_lim = ac_lim
        self.ac_dim = ac_dim
        self.obs_dim = obs_dim
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, ac_dim)
        self.log_scale = nn.Parameter(
            -1.34 * torch.ones(self.ac_dim), requires_grad=True
        )

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def act(self, obs, deterministic=False):
        action_mean = self.forward(obs)
        action_mean = action_mean * self.ac_lim
        normal = Normal(action_mean, torch.exp(self.log_scale))
        dist = Independent(normal, 1)
        if deterministic:
            action = action_mean.detach()
        else:
            action = dist.rsample()
        action_logprobs = dist.log_prob(torch.squeeze(action))

        return action, action_logprobs

    def get_actions_dist(self, obs):
        action_mean = self.forward(obs)
        action_mean = action_mean * self.ac_lim
        normal = Normal(action_mean, torch.exp(self.log_scale))
        dist = Independent(normal, 1)
        return dist


class AcM(nn.Module):
    def __init__(self, in_dim: int, ac_dim: int, ac_lim: int, discrete: bool = True):
        super().__init__()
        self.ac_lim = ac_lim

        self.discrete = discrete
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, ac_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        if self.discrete:
            x = torch.softmax(self.fc3(x), dim=1)
        else:
            x = torch.tanh(self.fc3(x))
            x = x * self.ac_lim
        return x

    def act(self, obs):
        action = self.forward(obs)
        if self.discrete:
            action = torch.argmax(action, dim=1)
        return action
