import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):

    def __init__(self, state_size: int, action_size: int, max_action: float, l1=400, l2=300):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, action_size),
            nn.Tanh()
        )
        self.reset_parameters()
        self.max_action = max_action

    def reset_parameters(self):
        self.net[0].weight.data.uniform_(*hidden_init(self.net[0]))
        self.net[2].weight.data.uniform_(*hidden_init(self.net[2]))
        self.net[4].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state) -> torch.Tensor:
        assert isinstance(state, torch.Tensor), "State is not of type torch.Tensor in ACTOR."
        assert state.dtype == torch.float32, "Tensor elements are not of type torch.float32 in ACTOR."
        assert state.shape[0] <= 24 or state.shape[0] >= BATCH_SIZE, "The tensor shape is not torch.Size([24]) in ACTOR."
        assert str(state.device.type) == str(DEVICE), "The state must be on the same device in ACTOR."

        # x = self.net(state)
        # action = self.max_action * x
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, l1=400, l2=300):
        super(Critic, self).__init__()

        # Critic Q1
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, 1)
        )

        # Critic Q2
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.net1[0].weight.data.uniform_(*hidden_init(self.net1[0]))
        self.net1[2].weight.data.uniform_(*hidden_init(self.net1[2]))
        self.net1[4].weight.data.uniform_(-3e-3, 3e-3)

        self.net2[0].weight.data.uniform_(*hidden_init(self.net2[0]))
        self.net2[2].weight.data.uniform_(*hidden_init(self.net2[2]))
        self.net2[4].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(state, torch.Tensor), "State is not of type torch.Tensor in CRITIC."
        assert state.dtype == torch.float32, "Tensor elements are not of type torch.float32 in CRITIC."
        assert state.shape[0] == BATCH_SIZE, "The tensor shape is not torch.Size([100]) in CRITIC."
        assert str(state.device.type) == str(DEVICE), "The state must be on the same device in CRITIC."

        assert isinstance(action, torch.Tensor), "Action is not of type torch.Tensor in CRITIC."
        assert action.dtype == torch.float32, "Tensor elements are not of type torch.float32 in CRITIC."
        assert action.shape[0] == BATCH_SIZE, "The action shape is not torch.Size([100]) in CRITIC."
        assert str(action.device.type) == str(DEVICE), "The action must be on the same device in CRITIC."

        sa = torch.cat([state, action], dim=1)

        return self.net1(sa), self.net2(sa)

    def Q1(self, state, action) -> torch.Tensor:
        assert isinstance(state, torch.Tensor), "State is not of type torch.Tensor in CRITIC."
        assert state.dtype == torch.float32, "Tensor elements are not of type torch.float32 in CRITIC."
        assert state.shape[0] == BATCH_SIZE, "The tensor shape is not torch.Size([100]) in CRITIC."
        assert str(state.device.type) == str(DEVICE), "The state must be on the same device in CRITIC."

        assert isinstance(action, torch.Tensor), "Action is not of type torch.Tensor in CRITIC."
        assert action.dtype == torch.float32, "Tensor elements are not of type torch.float32 in CRITIC."
        assert action.shape[0] == BATCH_SIZE, "The action shape is not torch.Size([100]) in CRITIC."
        assert str(action.device.type) == str(DEVICE), "The action must be on the same device in CRITIC."

        sa = torch.cat([state, action], dim=1)

        return self.net1(sa)
