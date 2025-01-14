from typing import Tuple

import numpy as np
from numba import njit
from numba.np.extensions import cross2d


@njit
def distance_to_goal(x: float, y: float, goal_x: float, goal_y: float) -> float:
    return np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)


@njit(fastmath=True, cache=True)
def angle_reward(
    x: float,
    y: float,
    theta: float,
    goal_x: float,
    goal_y: float,
    fov: float = 0.3,
    min_val: float = -1.0,
    max_val: float = 1.0,
) -> float:
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])

    cross_p = cross2d(o_t, g_t)
    dot_p = np.dot(o_t, g_t)
    alpha = np.abs(np.arctan2(np.abs(cross_p), dot_p))

    if alpha <= fov:
        return max_val
    else:
        return min_val


def angle_to_goal(
    x: float, y: float, theta: float, goal_x: float, goal_y: float
) -> float:
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])

    cross_product = np.cross(o_t, g_t)
    dot_product = np.dot(o_t, g_t)

    alpha = np.abs(np.arctan2(np.linalg.norm(cross_product), dot_product))

    # if alpha <= 0.0:
    #     alpha = 0.1
    # elif alpha >= 3.2:
    #     alpha = 3.3

    return alpha


def min_laser(measurement: np.ndarray, threshold: float):
    laser = np.min(measurement)
    if laser <= (threshold - 0.20):
        return True, laser
    else:
        return False, laser


@njit
def r3(x):
    if x < 0.3:
        return 1.0 - x
    else:
        return 0.0


def get_reward_improved(
    measurement: np.ndarray,
    distance: float,
    collision: bool,
    alpha: float,
    distance_init: float,
    step: int,
    time_penalty: float,
    threshold: float,
) -> Tuple[float, bool]:
    """
    measurement: array de leituras de laser
    distance: distância atual até o objetivo
    collision: se houve colisão
    alpha: ângulo em rdianos até o objetivo
    distance_init: distância inicial até o objetivo (pode ser do episódio ou do step anterior)
    step: contador de steps
    time_penalty: penalização por tempo

    Retorna (reward, done)
    """
    reward = -time_penalty

    if distance < threshold:
        return 100.0, True  # grande bônus ao chegar
    if collision:
        return -100.0, True  # grande punição se colidir

    progress = distance_init - distance
    reward += 5.0 * progress  #

    # Bônus de alinhamento (alpha pequeno é melhor)
    # Normaliza alpha de 0 (pouco alinhado) a 1 (bem alinhado)
    alpha_norm = 1.0 - (alpha / np.pi)
    if alpha_norm < 0:
        alpha_norm = 0.0
    elif alpha_norm > 1:
        alpha_norm = 1.0
    reward += 2.0 * alpha_norm

    # Penaliza obstáculo muito próximo
    obstacle = r3(min(measurement))
    reward -= obstacle  # subtrai valor se laser detectar algo muito perto

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
