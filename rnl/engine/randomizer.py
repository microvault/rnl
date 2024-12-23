import numpy as np

class TargetPosition:
    def __init__(
        self,
        window_size: int = 100,
        min_fraction: float = 0.1,
        max_fraction: float = 1.0,
        threshold: float = 0.01,
        adjustment: float = 0.05
    ):
        self.window_size = window_size
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.threshold = threshold
        self.adjustment = adjustment

    def adjust_initial_distance(
        self,
        reward_history,
        current_fraction,
    ):
        if len(reward_history) < 2 * self.window_size:
            # Se não tiver histórico suficiente, não mexe na fração
            return current_fraction

        recent_avg = np.mean(reward_history[-self.window_size:])
        prev_avg = np.mean(reward_history[-2 * self.window_size:-self.window_size])
        improvement = recent_avg - prev_avg

        if improvement > self.threshold:
            # melhorou
            new_fraction = min(current_fraction + self.adjustment, self.max_fraction)
        elif improvement < -self.threshold:
            # piorou
            new_fraction = max(current_fraction - self.adjustment, self.min_fraction)
        else:
            # ficou estável
            new_fraction = current_fraction

        return new_fraction
