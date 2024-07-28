from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class ModelActor(nn.Module):

    def __init__(
        self,
        state_dim: int = 13,
        action_dim: int = 2,
        max_action: float = 1.0,
        l1: int = 400,
        l2: int = 300,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        super().__init__()

        self.state_size = state_dim
        self.device = device
        self.batch_size = batch_size

        self.l1 = nn.Linear(state_dim, l1)
        self.l2 = nn.Linear(l1, l2)
        self.l3 = nn.Linear(l2, action_dim)
        self.reset_parameters()

        self.max_action = max_action

    def reset_parameters(self):
        self.l1.weight.data.uniform_(*hidden_init(self.l1))
        self.l2.weight.data.uniform_(*hidden_init(self.l2))
        self.l3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            state, torch.Tensor
        ), "State is not of type torch.Tensor in ACTOR."
        assert (
            state.dtype == torch.float32
        ), "Tensor elements are not of type torch.float32 in ACTOR."
        assert (
            state.shape[0] <= self.state_size or state.shape[0] >= self.batch_size
        ), "The tensor shape is not torch.Size([24]) in ACTOR."
        assert (
            str(state.device.type) == self.device
        ), "The state must be on the same device in ACTOR."

        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        action = self.max_action * torch.tanh(self.l3(x))

        return action

    def add_parameter_noise(self, scalar=0.1):
        for layer in [self.l1, self.l2, self.l3]:
            layer.weight.data += torch.randn_like(layer.weight.data) * scalar
            if layer.bias is not None:
                layer.bias.data += torch.randn_like(layer.bias.data) * scalar


class ModelCritic(nn.Module):
    def __init__(
        self,
        state_dim: int = 13,
        action_dim: int = 2,
        l1: int = 400,
        l2: int = 300,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        super().__init__()

        self.device = device
        self.batch_size = batch_size

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

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(
            state, torch.Tensor
        ), "State is not of type torch.Tensor in CRITIC."
        assert (
            state.dtype == torch.float32
        ), "Tensor elements are not of type torch.float32 in CRITIC."
        assert (
            state.shape[0] == self.batch_size
        ), "The tensor shape is not torch.Size([100]) in CRITIC."
        assert (
            str(state.device.type) == self.device
        ), "The state must be on the same device  in CRITIC."

        assert isinstance(
            action, torch.Tensor
        ), "Action is not of type torch.Tensor in CRITIC."
        assert (
            action.dtype == torch.float32
        ), "Tensor elements are not of type torch.float32 in CRITIC."
        assert (
            action.shape[0] == self.batch_size
        ), "The action shape is not torch.Size([100]) in CRITIC"
        assert (
            str(action.device.type) == self.device
        ), "The action must be on the same device  in CRITIC."

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
        return (q1, q2)

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            state, torch.Tensor
        ), "State is not of type torch.Tensor in CRITIC."
        assert (
            state.dtype == torch.float32
        ), "Tensor elements are not of type torch.float32 in CRITIC."
        assert (
            state.shape[0] == self.batch_size
        ), "The tensor shape is not torch.Size([100]) in CRITIC."
        assert (
            str(state.device.type) == self.device
        ), "The state must be on the same device in CRITIC."

        assert isinstance(
            action, torch.Tensor
        ), "Action is not of type torch.Tensor in CRITIC."
        assert (
            action.dtype == torch.float32
        ), "Tensor elements are not of type torch.float32 in CRITIC."
        assert (
            action.shape[0] == self.batch_size
        ), "The action shape is not torch.Size([100]) in CRITIC."
        assert (
            str(action.device.type) == self.device
        ), "The action must be on the same device in CRITIC."

        s = torch.cat([state, action], dim=1)

        s1 = F.relu(self.l1(s))
        s1 = F.relu(self.l2(s1))
        a1 = F.relu(self.l3(action))
        s1 = s1 + a1
        q1 = self.l4(s1)
        return q1
