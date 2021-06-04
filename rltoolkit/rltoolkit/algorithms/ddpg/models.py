import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim: int, ac_lim: float, ac_dim: int):
        super().__init__()

        self.ac_dim = ac_dim
        self.ac_lim = ac_lim
        self.obs_dim = obs_dim

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, ac_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        action = action * self.ac_lim
        return action, None

    def act(self, obs, *args, **kwargs):
        del args, kwargs  # unused
        with torch.no_grad():
            action, _ = self.forward(obs)
            return action, _


class Critic(nn.Module):
    def __init__(self, obs_dim: int, ac_dim: int):
        super().__init__()
        self.ac_dim = ac_dim
        self.fc1 = nn.Linear(obs_dim + ac_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, ac):
        x = torch.cat((obs, ac), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.squeeze(x, -1)
