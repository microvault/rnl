from typing import Tuple

import gym
import numpy as np
from gym.spaces import Box


class CustomWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)
        self.env = env

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, info = self.env.step(action)
        if isinstance(obs, tuple):
            obs = np.array(obs[0])
        return obs, np.float32(reward), np.bool_(terminated), info

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = np.array(obs[0])
        return obs
