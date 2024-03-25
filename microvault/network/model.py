from copy import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(
        self, state_size, action_size, max_action=1, fc1_units=400, fc2_units=300
    ):

        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.max_action = max_action

    def forward(self, state) -> torch.Tensor:
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, l1=400, l2=300):
        super().__init__()

        self.layer_1 = nn.Linear(state_dim + action_dim, l1)
        self.layer_2_s = nn.Linear(l1, l2)
        self.layer_2_a = nn.Linear(action_dim, l2)
        self.layer_3 = nn.Linear(l2, 1)

        self.layer_4 = nn.Linear(state_dim + action_dim, l1)
        self.layer_5_s = nn.Linear(l1, l2)
        self.layer_5_a = nn.Linear(action_dim, l2)
        self.layer_6 = nn.Linear(l2, 1)

    def forward(self, s, a) -> torch.Tensor:
        s1 = F.relu(self.layer_1(torch.cat([s, a], dim=1)))
        s1 = F.relu(self.layer_2_s(s1))
        a1 = F.relu(self.layer_2_a(a))
        s1 = s1 + a1
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(torch.cat([s, a], dim=1)))
        s2 = F.relu(self.layer_5_s(s2))
        a2 = F.relu(self.layer_5_a(a))
        s2 = s2 + a2
        q2 = self.layer_6(s2)

        return q1, q2
