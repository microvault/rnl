import numpy as np
from numba import njit
from typing import Tuple
import time
# ---------------------------------------------------
# Funções auxiliares (evite repetir)
# ---------------------------------------------------
@njit
def normalize_module(value, min_val, max_val, min_out, max_out):
    return min_out + (value - min_val) * (max_out - min_out) / (max_val - min_val)

@njit
def distance_to_goal(x: float, y: float, goal_x: float, goal_y: float) -> float:
    return np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)

@njit
def angle_to_goal(x: float, y: float, theta: float, goal_x: float, goal_y: float) -> float:
    """
    Retorna ângulo absoluto entre a orientação do robô e o objetivo.
    """
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])
    cross_p = np.cross(o_t, g_t)
    dot_p = np.dot(o_t, g_t)
    alpha = np.abs(np.arctan2(np.linalg.norm(cross_p), dot_p))
    return alpha

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

# ---------------------------------------------------
# Módulos de Recompensa
# ---------------------------------------------------
@njit
def collision_and_target_reward(
    distance: float,
    threshold: float,
    collision: bool,
    scale_collision: float = 1.0,
    scale_target: float = 1.0,
    reward_collision: float = -100.0,
    reward_target: float = 100.0
) -> Tuple[float, bool]:
    """
    1. Recompensa baseada em colisão e alvo.
    """
    # Se chegou no alvo
    if distance < threshold:
        return scale_target * reward_target, True
    # Se colidiu
    if collision:
        return scale_collision * reward_collision, True
    return 0.0, False

@njit
def orientation_reward(
    alpha: float,
    scale_orientation: float = 1.0
) -> float:
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
def global_progress_reward(
    distance_init: float,
    distance: float,
    scale_global: float = 1.0
) -> float:
    """
    Recompensa global baseada apenas no progresso (distância até o alvo).
    """
    progress = distance_init - distance
    reward = 5.0 * progress
    return scale_global * reward

@njit
def time_and_collision_reward(
    step: int,
    time_penalty: float,
    scale_time: float = 1.0
) -> float:
    """
    5. Recompensa por tempo e colisão (aqui focado no tempo).
    """
    # Exemplo: penalizar a cada passo
    return -scale_time * time_penalty

# ---------------------------------------------------
# Função principal de recompensa que unifica tudo
# ---------------------------------------------------

def get_reward(
    distance: float,
    collision: bool,
    alpha: float,
    distance_init: float,
    step: int,
    time_penalty: float,
    threshold: float,
    scale_collision: float = 1.0,
    scale_target: float = 1.0,
    scale_orientation: float = 1.0,
    scale_deadend: float = 1.0,
    scale_global: float = 1.0,
    scale_time: float = 1.0,
) -> Tuple[float, bool]:
    """
    measurement: leituras de laser
    distance: distância atual até o objetivo
    collision: se houve colisão
    alpha: ângulo até o objetivo
    distance_init: distância no início do episódio ou step anterior
    step: contador de steps
    time_penalty: penalização base por tempo
    threshold: raio para considerar que chegou no alvo
    scale_*: fatores de escala para cada módulo
    """
    done = False

    rew_coll_target, done_coll_target = collision_and_target_reward(
        distance, threshold, collision,
        scale_collision, scale_target
    )
    if done_coll_target:
        collision_score = normalize_module(rew_coll_target, -100, 100, -1, 1)
        return rew_coll_target, True

    orientation_rewards = orientation_reward(alpha, scale_orientation)

    progress_reward = global_progress_reward(
        distance_init, distance, scale_global
    )

    time_reward = time_and_collision_reward(step, time_penalty, scale_time)

    collision_score = normalize_module(rew_coll_target, -100, 100, -1, 1)
    orientation_score = normalize_module(orientation_rewards, 0, 1, -3, 3) # 30%
    progress_score = normalize_module(progress_reward, 0, 1, -5, 5) # 50%
    time_score = normalize_module(time_reward, -2, 0, -2.5, 0) # 20%

    reward = collision_score + orientation_score + progress_score + time_score


    print("----------------")
    print("Orientation:", orientation_score)
    print("----------------")
    print("Progress:", progress_score)
    print("----------------")
    print("Time:", time_score)
    print("----------------")
    print("Reward:", reward)

    return reward, done
