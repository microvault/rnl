import copy
import random
import numpy as np

class DomainRandomization:
    def __init__(
        self,
        window_size: int = 100,
    ):
        self.window_size = window_size

    def adjust_initial_distance(self,reward_history, current_fraction, min_fraction, max_fraction, threshold=0.01, adjustment=0.05):
        """
        Adjusts the initial distance fraction between the robot and the goal based on the agent's performance.
        
        reward_history: list of rewards per episode.
        current_fraction: current fraction of the maximum distance.
        min_fraction: minimum allowed fraction.
        max_fraction: maximum allowed fraction.
        threshold: minimum difference in average to consider improvement.
        adjustment: value to increase or decrease the fraction.
        """
        if len(reward_history) < 2 * self.window_size:
            return current_fraction
        recent_avg_reward = np.mean(reward_history[-self.window_size:])
        previous_avg_reward = np.mean(reward_history[-2*self.window_size:-self.window_size])
        improvement = recent_avg_reward - previous_avg_reward
        if improvement > threshold:
            # Agent is improving, increase the distance fraction
            new_fraction = min(current_fraction + adjustment, max_fraction)
        elif improvement < -threshold:
            # Performance worsened, decrease the distance fraction
            new_fraction = max(current_fraction - adjustment, min_fraction)
        else:
            # Stable performance, keep the current fraction
            new_fraction = current_fraction
        return new_fraction