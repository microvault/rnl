from typing import Tuple, Union

import gym
import numpy as np
from gym import spaces
from gym.spaces import Box


class CustomWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, int, np.ndarray],
        max_action: Union[float, int, np.ndarray],
    ):
        """Initializes the :class:`RescaleAction` wrapper.
        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        assert isinstance(
            env.action_space, spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        super().__init__(env)
        self.min_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + min_action
        )
        self.max_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + max_action
        )
        self.action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )
        low = self.observation_space.low[:24]
        high = self.observation_space.high[:24]
        self.observation_space = Box(low, high, dtype=np.float32)

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, info = self.env.step(action)
        obs = obs[:24]
        if isinstance(obs, tuple):
            obs = np.array(obs[0])
        return obs, np.float32(reward), np.bool_(terminated), info

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        obs = obs[:24]
        if isinstance(obs, tuple):
            obs = np.array(obs[0])
        return obs

    def action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.
        Args:
            action: The action to rescale
        Returns:
            The rescaled action
        """
        assert np.all(np.greater_equal(action, self.min_action)), (
            action,
            self.min_action,
        )
        assert np.all(np.less_equal(action, self.max_action)), (action, self.max_action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = np.clip(action, low, high)
        return action
