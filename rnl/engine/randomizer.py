import numpy as np

class TargetPosition:
    def __init__(
        self,
        window_size: int,
        min_fraction: float,
        max_fraction: float,
        threshold: float,
        adjustment: float,
        episodes_interval: int
    ):
        self.window_size = window_size
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.threshold = threshold
        self.adjustment = adjustment
        self.episodes_interval = episodes_interval

    def adjust_initial_distance(
        self,
        reward_history,
        current_fraction,
        epoch
    ):
        if epoch % self.episodes_interval != 0:
            return current_fraction

        if len(reward_history) < 2 * self.window_size:
            return current_fraction

        recent_avg = np.mean(reward_history[-self.window_size:])
        prev_avg = np.mean(reward_history[-2 * self.window_size:-self.window_size])
        improvement = recent_avg - prev_avg

        if improvement > self.threshold:
            new_fraction = min(current_fraction + self.adjustment, self.max_fraction)
        elif improvement < -self.threshold:
            new_fraction = max(current_fraction - self.adjustment, self.min_fraction)
        else:
            new_fraction = current_fraction

        return new_fraction
