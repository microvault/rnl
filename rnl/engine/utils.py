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

    cross_product = np.cross(o_t, g_t)
    dot_product = np.dot(o_t, g_t)

    alpha = np.abs(np.arctan2(np.linalg.norm(cross_product), dot_product))

    if alpha <= 0.0:
        alpha = 0.1
    elif alpha >= 3.2:
        alpha = 3.3

    return alpha


def min_laser(measurement: np.ndarray, threshold: float):
    laser = np.min(measurement)
    if laser <= (threshold - 0.20):
        return True, laser
    else:
        return False, laser


def get_reward(
    measurement, distance, collision, alpha, distance_init
) -> Tuple[float, bool]:

    reward = 0.0

    alpha_norm = (1 / alpha) * 0.2

    if alpha_norm >= 0.0 and alpha_norm <= 0.7:
        alpha_norm = 2
    else:
        alpha_norm = 0

    if distance < 0.2:
        return 500.0, True
    elif collision:
        return -500.0, True
    else:
        obstacle = r3(min(measurement))
        if distance < 0.4:
            obstacle = 0

        reward = distance_init + abs(alpha_norm) - abs(distance) - obstacle
        return reward, False


@njit
def r3(x):
    if x < 0.3:
        return 1 - x
    else:
        return 0.0


@njit
def uniform_random(min_val, max_val):
    return np.random.uniform(min_val, max_val)


@njit
def uniform_random_int(min_val, max_val):
    return np.random.randint(min_val, max_val + 1)
