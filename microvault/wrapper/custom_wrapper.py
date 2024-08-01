from typing import Tuple

import gym
import numpy as np
from gym.spaces import Box


class CustomWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
    ):
        """Initializes the :class:`RescaleAction` wrapper.
        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """

        super().__init__(env)
        low = self.observation_space.low[:13]
        high = self.observation_space.high[:13]
        self.observation_space = Box(low, high, dtype=np.float32)

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, info = self.env.step(action)
        obs = obs[:13]
        if isinstance(obs, tuple):
            obs = np.array(obs[0])
        return obs, np.float32(reward), np.bool_(terminated), info

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        obs = obs[:13]
        if isinstance(obs, tuple):
            obs = np.array(obs[0])
        return obs
