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

    def __init__(self, state_size, action_size, max_action, l1=400, l2=300):
        super().__init__()

        self.l1 = nn.Linear(state_size, l1)
        self.l2 = nn.Linear(l1, l2)
        self.l3 = nn.Linear(l2, action_size)
        self.reset_parameters()

        self.max_action = max_action

    def reset_parameters(self):
        self.l1.weight.data.uniform_(*hidden_init(self.l1))
        self.l2.weight.data.uniform_(*hidden_init(self.l2))
        self.l3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state) -> torch.Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        action = self.max_action * torch.tanh(self.l3(x))
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, l1=400, l2=300):
        super().__init__()

        self.l1 = nn.Linear(state_dim + action_dim, l1)
        self.l2 = nn.Linear(l1, l2)
        self.l3 = nn.Linear(action_dim, l2)
        self.l4 = nn.Linear(l2, 1)
        self.reset_parameters_q1()

        self.l5 = nn.Linear(state_dim + action_dim, l1)
        self.l6 = nn.Linear(l1, l2)
        self.l7 = nn.Linear(action_dim, l2)
        self.l8 = nn.Linear(l2, 1)
        self.reset_parameters_q2()

    def reset_parameters_q1(self):
        self.l1.weight.data.uniform_(*hidden_init(self.l1))
        self.l2.weight.data.uniform_(*hidden_init(self.l2))
        self.l3.weight.data.uniform_(*hidden_init(self.l3))
        self.l4.weight.data.uniform_(-3e-3, 3e-3)

    def reset_parameters_q2(self):
        self.l5.weight.data.uniform_(*hidden_init(self.l5))
        self.l6.weight.data.uniform_(*hidden_init(self.l6))
        self.l7.weight.data.uniform_(*hidden_init(self.l7))
        self.l8.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action) -> torch.Tensor:
        s = torch.cat([state, action], dim=1)

        s1 = F.relu(self.l1(s))
        s1 = F.relu(self.l2(s1))
        a1 = F.relu(self.l3(action))
        s1 = s1 + a1
        q1 = self.l4(s1)

        s2 = F.relu(self.l5(s))
        s2 = F.relu(self.l6(s2))
        a2 = F.relu(self.l7(action))
        s2 = s2 + a2
        q2 = self.l8(s2)
        return q1, q2

    def Q1(self, state, action) -> torch.Tensor:
        s = torch.cat([state, action], dim=1)

        s1 = F.relu(self.l1(s))
        s1 = F.relu(self.l2(s1))
        a1 = F.relu(self.l3(action))
        s1 = s1 + a1
        q1 = self.l4(s1)
        return q1