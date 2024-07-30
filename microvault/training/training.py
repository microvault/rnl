import timeit
from dataclasses import dataclass
from typing import Tuple

import gym
import numpy as np

from microvault.algorithms.agent import Agent
from microvault.components.replaybuffer import ReplayBuffer
from microvault.wrapper.custom_wrapper import CustomWrapper


@dataclass
class Training:
    def __init__(
        self,
        environment: gym.Env,
        agent: Agent,
        replaybuffer: ReplayBuffer,
        min_action: float,
        max_action: float,
    ):
        self.env = environment
        self.agent = agent
        self.replaybuffer = replaybuffer
        self.min_action = min_action
        self.max_action = max_action

    def train_one_epoch(
        self,
        batch_size: int,
        timestep: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
    ]:

        env = CustomWrapper(self.env, self.min_action, self.max_action)
        start_time = timeit.default_timer()
        state = env.reset()
        done = False
        score = 0
        critic_loss = 0.0
        actor_loss = 0.0
        q = 0.0
        max_q = 0.0
        intrinsic_reward = 0.0
        error = 0.0

        for t in range(timestep):
            action = self.agent.predict(state)
            next_state, reward, done, info = env.step(action)
            self.replaybuffer.add(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done or t == (timestep - 1):
                if len(self.replaybuffer) > batch_size:
                    critic_loss, actor_loss, q, max_q, intrinsic_reward, error = (
                        self.agent.learn(memory=self.replaybuffer, n_iteration=t + 1)
                    )
                break

        elapsed_time = timeit.default_timer() - start_time

        return (
            critic_loss,
            actor_loss,
            q,
            max_q,
            intrinsic_reward,
            error,
            score,
            elapsed_time,
        )
