from typing import Tuple

import numpy as np
from numba import njit
from shapely.geometry import Point


@njit
def normalize_module(value, min_val, max_val):
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value


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
        return -1, True
    if distance < threshold:
        return 1, True
    if collision:
        return -1, True
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
def global_progress_reward(
    distance_initial: float, current_distance: float, scale_distance: float
) -> float:
    diff = distance_initial - current_distance
    reward = (diff / 8) * scale_distance
    return reward


@njit
def r3(x: float, threshold_collision: float, scale: float) -> float:
    margin = 0.3
    if x <= threshold_collision:
        return -scale
    elif x < threshold_collision + margin:
        return -scale * (threshold_collision + margin - x) / margin
    else:
        return 0.0


def get_reward(
    type_reward: str,
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
) -> Tuple[float, float, float, float, float, bool]:
    done = False
    scale_orientation = 0.003
    scale_distance = 0.1
    scale_time = 0.01
    scale_obstacle = 0.001

    rew_coll_target, done_coll_target = collision_and_target_reward(
        current_distance, threshold, collision, position_x, position_y, poly
    )

    time_reward = time_and_collision_reward(step, 1.0, scale_time)

    orientation_rewards = orientation_reward(alpha, scale_orientation)

    obstacle_reward = r3(min(measurement), threshold_collision, scale_obstacle)

    progress_reward = global_progress_reward(
        initial_distance, current_distance, scale_distance
    )

    norm_progress_reward = normalize_module(
        progress_reward, -scale_distance, scale_distance
    )

    if done_coll_target:
        return rew_coll_target, 0.0, 0.0, 0.0, 0.0, True

    elif type_reward == "time":
        return rew_coll_target, 0.0, 0.0, time_reward, 0.0, done

    elif type_reward == "distance":
        return rew_coll_target, 0.0, norm_progress_reward, 0.0, 0.0, done

    elif type_reward == "orientation":
        return rew_coll_target, orientation_rewards, 0.0, 0.0, 0.0, done

    elif type_reward == "all":
        return (
            rew_coll_target,
            orientation_rewards,
            norm_progress_reward,
            time_reward,
            obstacle_reward,
            done,
        )

    elif type_reward == "any":
        return rew_coll_target, 0.0, 0.0, 0.0, 0.0, done

    elif type_reward == "distance_orientation":
        return (
            rew_coll_target,
            orientation_rewards,
            norm_progress_reward,
            0.0,
            0.0,
            done,
        )

    elif type_reward == "distance_time":
        return rew_coll_target, 0.0, norm_progress_reward, time_reward, 0.0, done

    elif type_reward == "orientation_time":
        return rew_coll_target, orientation_rewards, 0.0, time_reward, 0.0, done

    elif type_reward == "distance_orientation_time":
        return (
            rew_coll_target,
            orientation_rewards,
            norm_progress_reward,
            time_reward,
            0.0,
            done,
        )

    elif type_reward == "distance_obstacle":
        return rew_coll_target, 0.0, norm_progress_reward, 0.0, obstacle_reward, done

    elif type_reward == "orientation_obstacle":
        return rew_coll_target, orientation_rewards, 0.0, 0.0, obstacle_reward, done

    else:
        return rew_coll_target, 0.0, 0.0, 0.0, 0.0, done
