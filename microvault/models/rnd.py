import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RNDModel(nn.Module):
    def __init__(self, input_size):
        super(RNDModel, self).__init__()

        self.target = nn.Sequential(
            nn.Linear(input_size, 516),
            nn.ReLU(),
            nn.Linear(516, 516),
            nn.ReLU(),
            nn.Linear(516, 516),
        )

        self.predictor = nn.Sequential(
            nn.Linear(input_size, 516),
            nn.ReLU(),
            nn.Linear(516, 516),
            nn.ReLU(),
            nn.Linear(516, 516),
        )

        # for param in self.target.parameters():
        #     param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=0.0001)

    def calcule_reward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        # print(x)
        y_true = self.target(x).detach()
        y_pred = self.predictor(x)
        reward = torch.pow(y_pred - y_true, 2).sum()

        # if torch.isnan(reward):
        #     reward = torch.tensor(0.1, dtype=torch.float32)
        return reward

    def update(self, reward):
        # self.optimizer.zero_grad()
        reward.sum().backward()
        self.optimizer.step()
