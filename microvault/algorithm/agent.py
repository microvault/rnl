import copy
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from numpy import inf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self, env: gym.Env, replay_buffer: ReplayBuffer, max_action, min_action
    ) -> None:
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            max_action (ndarray): the maximum valid value for each action vector
            min_action (ndarray): the minimum valid value for each action vector
            noise (float): the range to generate random noise while learning
            noise_std (float): the range to generate random noise while performing action
            noise_clip (float): to clip random noise into this range
        """
        self.env = env
        self.reset()
        self.state = self.env.reset()

        self.max_action = max_action
        self.min_action = min_action

        # Replay memory
        self.memory = replay_buffer

    def action(self, actor, device) -> np.ndarray:
        """Returns actions for given state as per current policy."""
        state = torch.tensor([self.state])

        if device not in ["cpu"]:
            state = state.cuda(device).cpu().data.numpy()

        action = actor(state).cpu().data.numpy()

        action = action.clip(self.min_action[0], self.max_action[0])

        return action

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.state = self.env.reset()

    @torch.no_grad()
    def step(self, actor: nn.Module, device: str = "cpu") -> Tuple[float, bool]:
        action = self.action(actor, device)

        next_state, reward, done, _ = self.env.step(action[0])

        self.memory.add(self.state, action[0], reward, next_state, done)

        self.state = next_state

        if done:
            self.reset()
        return reward, done


env = CustomWrapper(gym.make("BipedalWalker-v3"), min_action=-1.0, max_action=1.0)
memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
agent = Agent(
    env=env,
    replay_buffer=memory,
    max_action=env.action_space.high,
    min_action=env.action_space.low,
)
actor = Actor(
    env.observation_space.shape[0],
    env.action_space.shape[0],
    float(env.action_space.high[0]),
).to(device)

for i in range(10):
    agent.step(actor)
