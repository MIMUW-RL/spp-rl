import torch
import numpy
from torch import nn

# best found simple feed-forward model for learning inverse dynamics,
# verified and tested on InvertedDoublePendulum, Hopper, HalfCheetah
#
# does not work well on Walker2d


class BasicAcM(nn.Module):
    def __init__(self, in_dim: int, ac_dim: int, discrete: bool = True):
        super().__init__()
        self.discrete = discrete
        h1s = 100
        h2s = 50
        self.fc1 = nn.Linear(in_dim, h1s)
        self.fc2 = nn.Linear(h1s, h2s)
        self.fc21 = nn.Linear(in_dim, h2s)
        self.fc3 = nn.Linear(h2s, ac_dim)
        self.t = nn.Parameter(torch.FloatTensor([1]))
        self.t1 = nn.Parameter(torch.FloatTensor(numpy.repeat(1, ac_dim)))

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        h1 = torch.tanh(self.fc2(h) + self.t * self.fc21(x))
        r = torch.tanh(self.fc3(h1)) * self.t1
        return r

    def act(self, obs):
        action = self.forward(obs)
        return action
