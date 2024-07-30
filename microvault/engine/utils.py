import math
from typing import Tuple

import numpy as np
from numba import njit


@njit
def distance_to_goal(x: float, y: float, goal_x: float, goal_y: float) -> float:
    return np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)


@njit
def angle_to_goal(
    x: float, y: float, theta: float, goal_x: float, goal_y: float
) -> float:
    skewX = goal_x - x
    skewY = goal_y - y
    dot = skewX * 1 + skewY * 0
    mag1 = math.sqrt(math.pow(skewX, 2) + math.pow(skewY, 2))
    mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
    beta = math.acos(dot / (mag1 * mag2))
    if skewY < 0:
        if skewX < 0:
            beta = -beta
        else:
            beta = 0 - beta
    beta2 = beta - theta
    if beta2 > np.pi:
        beta2 = np.pi - beta2
        beta2 = -np.pi - beta2
    if beta2 < -np.pi:
        beta2 = -np.pi - beta2
        beta2 = np.pi - beta2

    return beta2


def min_laser(measurement: np.ndarray, threshold: float = 0.1):
    laser = np.min(measurement)
    if laser < threshold:
        return True, laser
    else:
        return False, laser


def get_reward(distance, action, measurement, collision) -> Tuple[np.float64, np.bool_]:
    if distance < 0.3:
        return 80.0, np.bool_(True)
    elif collision:
        return -100.0, np.bool_(True)
    else:
        reward = action[0] / 2 - abs(action[1]) / 2 - r3(min(measurement)) / 2
        return np.float32(reward), np.bool_(False)


@njit
def r3(x):
    if x < 1:
        return 1 - x
    else:
        return 0.0
