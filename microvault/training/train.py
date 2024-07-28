import timeit
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import gym
import numpy as np

from microvault.algorithms.agent import Agent
from microvault.components.replaybuffer import ReplayBuffer


@dataclass
class Training:
    def __init__(
        self,
        env: gym.Env,
        agent: Agent,
        replaybuffer: ReplayBuffer,
    ):
        self.env = env
        self.agent = agent
        self.replaybuffer = replaybuffer

    def train_one_epoch(
        self,
        batch_size: int,
        timestep: int,
        scalar_deque: deque,
        scalar_decay_deque: deque,
        distance_deque: deque,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        deque,
        deque,
        deque,
        float,
    ]:
        start_time = timeit.default_timer()
        state = self.env.reset()
        done = False
        score = 0
        max_t = 0

        critic_loss = 0.0
        actor_loss = 0.0
        q = 0.0
        max_q = 0.0
        intrinsic_reward = 0.0
        error = 0.0

        for t in range(timestep):
            if isinstance(state, tuple):
                state = np.array(state[0])

            action, scalar, scalar_decay, distance, _, _ = self.agent.predict(state)
            next_state, reward, done, info = self.env.step(action)
            self.replaybuffer.add(state, action, reward, next_state, done)

            state = next_state
            score += reward

            scalar_deque.append(scalar)
            scalar_decay_deque.append(scalar_decay)
            distance_deque.append(distance)

            if done or t == (timestep - 1):
                if len(self.replaybuffer) > batch_size:
                    critic_loss, actor_loss, q, max_q, intrinsic_reward, error = (
                        self.agent.learn(
                            memory=self.replaybuffer, n_iteration=max_t + 1, episode=10
                        )
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
            scalar_deque,
            scalar_decay_deque,
            distance_deque,
            elapsed_time,
        )
