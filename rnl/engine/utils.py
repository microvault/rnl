from typing import Tuple

import numpy as np
from numba import njit


@njit
def distance_to_goal(x: float, y: float, goal_x: float, goal_y: float) -> float:
    return np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)


def angle_to_goal(
    x: float, y: float, theta: float, goal_x: float, goal_y: float
) -> float:
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])

    # Ângulo alpha
    cross_product = np.cross(o_t, g_t)
    dot_product = np.dot(o_t, g_t)

    alpha = np.abs(np.arctan2(np.linalg.norm(cross_product), dot_product))

    # Cálculo de alpha_norm
    if alpha <= 0.0:
        alpha = 0.1
    elif alpha >= 3.2:
        alpha = 3.3

    return alpha


def min_laser(measurement: np.ndarray, threshold: float = 0.1):
    laser = np.min(measurement)
    if laser < threshold:
        return True, laser
    else:
        return False, laser


def get_reward(
    measurement, distance, collision, alpha, distance_init
) -> Tuple[np.float64, np.bool_]:

    reward = 0.0

    alpha_norm = (1 / alpha) * 0.2

    if alpha_norm >= 0.0 and alpha_norm <= 0.7:
        alpha_norm = 2
    else:
        alpha_norm = 0

    if distance < 0.4:
        return 500.0, np.bool_(True)
    elif collision:
        return -500.0, np.bool_(True)
    else:
        obstacle = r3(min(measurement))
        if distance < 0.4:
            obstacle = 0

        reward = distance_init + abs(alpha_norm) - abs(distance) - obstacle
        return np.float32(reward), np.bool_(False)


@njit
def r3(x):
    if x < 0.3:
        return 1 - x
    else:
        return 0.0


@njit
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)
