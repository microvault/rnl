import copy
import random

import numpy as np


class OUNoise:

    def __init__(self, size: int, seed: int, mu: float, theta: float, sigma: float):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for i in range(len(x))]
        )
        self.state = x + dx
        return self.state
