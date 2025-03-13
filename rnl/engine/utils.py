import random

import numpy as np
import torch
from numba import njit


@njit
def normalize_module(value, min_val, max_val):
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value


@njit
def distance_to_goal(
    x: float, y: float, goal_x: float, goal_y: float, max_value: float
) -> float:
    dist = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
    if dist >= max_value:
        return max_value
    else:
        return dist


# @njit # !!!!!!
def angle_to_goal(
    x: float, y: float, theta: float, goal_x: float, goal_y: float, max_value: float
) -> float:
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])

    cross_product = np.cross(o_t, g_t)
    dot_product = np.dot(o_t, g_t)

    alpha = np.abs(np.arctan2(np.linalg.norm(cross_product), dot_product))

    if alpha >= max_value:
        return max_value

    else:
        return alpha


@njit
def min_laser(measurement: np.ndarray, threshold: float):
    laser = np.min(measurement)
    if laser <= threshold:
        return True, laser
    else:
        return False, laser


@njit
def uniform_random(min_val, max_val):
    return np.random.uniform(min_val, max_val)


@njit
def uniform_random_int(min_val, max_val):
    return np.random.randint(min_val, max_val + 1)


def safe_stats(data):
    clean_data = [v for v in data if v is not None]
    if not clean_data:
        return 0.0, 0.0, 0.0
    return np.mean(clean_data), np.min(clean_data), np.max(clean_data)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_info(info: dict) -> dict:
    """Remove as chaves indesejadas do dicionário de info."""
    keys_to_remove = [
        "final_observation",
        "_final_observation",
        "final_info",
        "_final_info",
    ]
    return {k: v for k, v in info.items() if k not in keys_to_remove}


def statistics(info_list, field):
    values = [info[field] for info in info_list if field in info]
    if not values:  # se a lista estiver vazia, retorne None ou valores padrão
        return None, None, None, None
    mean_value = np.mean(values)
    min_value = np.min(values)
    max_value = np.max(values)
    std_deviation = np.std(values)
    return mean_value, min_value, max_value, std_deviation

def calculate_batch_nsteps(n_env: int, desired_batch_total: int):
    """
    Calculates and returns the total batch size and n_steps ensuring:
      total_batch = n_steps * n_env
    If the desired_batch_total is not divisible by n_env,
    it will be adjusted to the nearest lower multiple.

    Parameters:
      n_env (int): Number of environments.
      desired_batch_total (int): Desired total batch size (should be large).

    Returns:
      total_batch (int): Adjusted total batch size.
      n_steps (int): Number of steps per environment.
    """
    # Adjust desired_batch_total to be divisible by n_env
    if desired_batch_total % n_env != 0:
        total_batch = (desired_batch_total // n_env) * n_env
    else:
        total_batch = desired_batch_total

    n_steps = total_batch // n_env
    return total_batch, n_steps


# Example usage:
n_env = 4
desired_batch_total = 1024  # for example
total_batch, n_steps = calculate_batch_nsteps(n_env, desired_batch_total)
print(f"Total batch: {total_batch}, n_steps per environment: {n_steps}")
