from typing import Tuple

import numpy as np
from numba import njit
from shapely.geometry import Point


@njit
def normalize_module(value, min_val, max_val, min_out, max_out):
    return min_out + (value - min_val) * (max_out - min_out) / (max_val - min_val)


@njit
def distance_to_goal(x: float, y: float, goal_x: float, goal_y: float) -> float:
    return np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)


# @njit
# def angle_to_goal(
#     x: float, y: float, theta: float, goal_x: float, goal_y: float
# ) -> float:
#     """
#     Retorna ângulo absoluto entre a orientação do robô e o objetivo.
#     """
#     o_t = np.array([np.cos(theta), np.sin(theta)])
#     g_t = np.array([goal_x - x, goal_y - y])
#     cross_p = np.cross(o_t, g_t)
#     dot_p = np.dot(o_t, g_t)
#     alpha = np.abs(np.arctan2(np.linalg.norm(cross_p), dot_p))
#     return alpha
#
@njit
def angle_to_goal(x, y, theta, goal_x, goal_y):
    # Ângulo do objetivo em relação ao (0,0)
    direction = np.arctan2(goal_y - y, goal_x - x)
    # Diferença entre a orientação atual e a direção do objetivo
    alpha = direction - theta
    # Normaliza pra ficar em [-π, π]
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    return abs(alpha)

@njit
def r3(x: float, threshold) -> float:
    """
    Penaliza quando o obstáculo está a menos de 0.3m.
    """
    if x < threshold:
        return 1.0 - x
    else:
        return 0.0


@njit
def min_laser(measurement: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """
    Retorna se há obstáculo muito próximo e o valor do laser mais próximo.
    """
    laser = np.min(measurement)
    return (laser <= (threshold - 0.20)), laser


def collision_and_target_reward(
    distance: float,
    threshold: float,
    collision: bool,
    x: float,
    y: float,
    poly,
) -> Tuple[float, bool]:
    """
    1. Recompensa baseada em colisão e alvo.
    """
    if not poly.contains(Point(x, y)):
        return -10, True
    if distance < threshold:
        return 10, True
    if collision:
        return -10, True
    return 0.0, False


@njit
def orientation_reward(alpha: float, scale_orientation: float = 1.0) -> float:
    """
    2. Recompensa de orientação (quanto menor o alpha, melhor).
    """
    alpha_norm = 1.0 - (alpha / np.pi)
    if alpha_norm < 0.0:
        alpha_norm = 0.0
    elif alpha_norm > 1.0:
        alpha_norm = 1.0
    return scale_orientation * alpha_norm


@njit
def time_and_collision_reward(
    step: int, time_penalty: float, scale_time: float = 1.0
) -> float:
    """
    5. Recompensa por tempo e colisão (aqui focado no tempo).
    """
    return -scale_time * time_penalty


@njit
def global_progress_reward(distance: float, scale: float) -> float:
    min_d = 4.0
    max_d = 63.57
    reward = ((max_d - distance) / (max_d - min_d)) * 10 - 5
    reward *= scale
    reward = max(min(reward, 5), -5)
    return reward


def get_reward(
    poly,
    position_x: float,
    position_y: float,
    distance: float,
    collision: bool,
    alpha: float,
    step: int,
    time_penalty: float,
    threshold: float,
    scale_orientation: float,
    scale_distance: float,
    scale_time: float,
) -> Tuple[float, float, float, float, bool]:
    done = False
    rew_coll_target, done_coll_target = collision_and_target_reward(
        distance, threshold, collision, position_x, position_y, poly
    )

    time_reward = time_and_collision_reward(step, time_penalty, scale_time)

    # orientation_rewards = orientation_reward(alpha, scale_orientation)
    # orientation_score = normalize_module(orientation_rewards, 0, 1, -3, 3)  # 30%
    # time_score = normalize_module(time_reward, -1, 0, -1, 0)  # 20%
    # progress_reward = global_progress_reward(distance, scale_distance)  # 50%

    if done_coll_target:
        return rew_coll_target, 0.0, 0.0, 0.0, True

    return rew_coll_target, 0.0, 0.0, time_reward, done
