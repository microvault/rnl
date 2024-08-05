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
    ):
        self.env = environment
        self.agent = agent
        self.replaybuffer = replaybuffer

    def train_one_epoch(
        self,
        batch_size: int,
        timestep: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:

        env = CustomWrapper(self.env)
        start_time = timeit.default_timer()
        state = env.reset()
        done = False
        score = 0
        model_loss = 0.0
        q = 0.0
        max_q = 0.0
        selection_action = 0.0
        alpha = 0.0
        beta = 0.0
        total_priority = 0.0
        data_length = 0.0
        total_priorities = 0.0
        min_priorities = 0.0
        max_priorities = 0.0
        mean_priorities = 0.0

        for t in range(timestep):
            action = self.agent.predict(state)
            next_state, reward, done, info = env.step(action)
            self.replaybuffer.add(state, action, reward, next_state, done)

            state = next_state
            score += reward
            selection_action = action

            print(
                "\rTimestep {:.2f}\tActions: {:.2f}\tReward: {:.2f}".format(
                    t, action, reward
                ),
                end="",
            )

            if done or t == (timestep - 1):
                if len(self.replaybuffer) > batch_size:
                    (
                        model_loss,
                        q,
                        max_q,
                        alpha,
                        beta,
                        total_priority,
                        data_length,
                        total_priorities,
                        min_priorities,
                        max_priorities,
                        mean_priorities,
                    ) = self.agent.learn(memory=self.replaybuffer, n_iteration=t + 1)
                break

        elapsed_time = timeit.default_timer() - start_time

        return (
            model_loss,
            q,
            max_q,
            selection_action,
            score,
            elapsed_time,
            alpha,
            beta,
            total_priority,
            data_length,
            total_priorities,
            min_priorities,
            max_priorities,
            mean_priorities,
        )
