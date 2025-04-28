from typing import Any, Dict, Tuple

import numpy as np
from numba import njit
from shapely.geometry import Point


def collision_and_target_reward(
    distance: float, threshold: float, collision: bool, x: float, y: float, poly
) -> Tuple[float, bool]:
    """
    Calcula a recompensa baseada na proximidade do alvo e colisões.

    Retorna:
      - reward: Recompensa (1 para sucesso, -1 para colisão ou fora da área, 0 caso contrário)
      - done: Booleano indicando se a condição de término foi atingida.
    """
    if not poly.contains(Point(x, y)):
        return -1.0, True
    if distance < threshold:
        return 1.0, True
    if collision:
        return -1.0, True
    return 0.0, False


@njit
def orientation_reward(alpha: float, scale_orientation: float) -> float:
    alpha_norm = 1.0 - (alpha / np.pi)
    if alpha_norm < 0.0:
        alpha_norm = 0.0
    elif alpha_norm > 1.0:
        alpha_norm = 1.0

    return scale_orientation * alpha_norm - scale_orientation


@njit
def time_and_collision_reward(scale_time: float = 0.01) -> float:
    return -scale_time  # * 0.01


@njit
def prog_reward(
    current_distance: float,
    min_distance: float,
    max_distance: float,
    scale_factor: float,
) -> float:

    # A recompensa é negativa: quanto maior o normalized, mais longe está e maior a penalização.
    reward = -scale_factor * current_distance
    return reward


# Função 5: Recompensa baseada em obstáculos (usando o valor mínimo medido)
@njit
def r3(x: float, threshold_collision: float, scale: float) -> float:
    margin = 0.3
    if x <= threshold_collision:
        return -scale
    elif x < threshold_collision + margin:
        return -scale * (threshold_collision + margin - x) / margin
    else:
        return 0.0


class RewardConfig:
    def __init__(self, params: Dict[str, Any]):
        """
        Parâmetros:
          - params: Dicionário com os parâmetros, como escalas:
              * scale_orientation, scale_distance, scale_time, scale_obstacle.
        """
        self.params = params

    def get_reward(
        self,
        measurement,
        poly,
        position_x: float,
        position_y: float,
        initial_distance: float,
        current_distance: float,
        collision: bool,
        alpha: float,
        step: int,
        threshold: float,
        threshold_collision: float,
        min_distance: float,
        max_distance: float,
    ) -> Tuple[float, float, float, float, float, bool]:
        scale_time = self.params.get("scale_time", 0.01)
        scale_distance = self.params.get("scale_distance", 0.1)
        scale_orientation = self.params.get("scale_orientation", 0.003)
        scale_obstacle = self.params.get("scale_obstacle", 0.001)

        rew_coll_target, done_coll_target = collision_and_target_reward(
            current_distance, threshold, collision, position_x, position_y, poly
        )
        time_reward = time_and_collision_reward(scale_time)
        orient_reward = orientation_reward(alpha, scale_orientation)
        obstacle_reward = r3(min(measurement), threshold_collision, scale_obstacle)
        progress_reward = prog_reward(
            current_distance,
            min_distance,
            max_distance,
            scale_distance,
        )
        if done_coll_target:
            return rew_coll_target, 0.0, 0.0, 0.0, 0.0, True

        else:
            return (
                rew_coll_target,
                orient_reward,
                progress_reward,
                time_reward,
                obstacle_reward,
                False,
            )
