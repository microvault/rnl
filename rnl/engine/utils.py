import numpy as np
from numba import njit


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
    x: float, y: float, theta: float, goal_x: float, goal_y: float
) -> float:
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])

    cross_product = np.cross(o_t, g_t)
    dot_product = np.dot(o_t, g_t)

    alpha = np.abs(np.arctan2(np.linalg.norm(cross_product), dot_product))

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


def plot_metrics(metrics, completed_rewards, completed_lengths):
    import matplotlib.pyplot as plt

    step_range = range(1, len(metrics["total_reward"]) + 1)
    step_metrics = [
        ("Obstacles Score", metrics["obstacles"], "brown"),
        ("Collision Score", metrics["collision"], "red"),
        ("Orientation Score", metrics["orientation"], "green"),
        ("Progress Score", metrics["progress"], "blue"),
        ("Time Score", metrics["time"], "orange"),
        ("Total Reward", metrics["total_reward"], "purple"),
        ("Action", metrics["action"], "blue"),
        ("Distance", metrics["distance"], "cyan"),
        ("Alpha", metrics["alpha"], "magenta"),
        ("Min Lidar", metrics["min_lidar"], "yellow"),
        ("Max Lidar", metrics["max_lidar"], "black"),
        ("Steps to Goal", metrics["steps_to_goal"], "purple"),
        ("Steps Below Threshold", metrics["steps_below_threshold"], "magenta"),
        ("Steps to Collision", metrics["steps_to_collision"], "orange"),
        ("Turns Left", metrics["turn_left_count"], "cyan"),
        ("Turns Right", metrics["turn_right_count"], "lime"),
    ]

    total_plots = len(step_metrics) + 1
    cols = 2
    rows = (total_plots + cols - 1) // cols
    plt.figure(figsize=(10, 5 * rows))

    for idx, (title, data, color) in enumerate(step_metrics, start=1):
        ax = plt.subplot(rows, cols, idx)
        ax.plot(
            step_range, data, label=title, color=color, linestyle="-", linewidth=1.5
        )
        ax.set_ylabel(title, fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="x", labelbottom=False, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        mean_val, min_val, max_val = safe_stats(data)
        ax.text(
            0.5,
            -0.25,
            f"Média: {mean_val:.4f} | Min: {min_val:.4f} | Max: {max_val:.4f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=6,
        )

    # Plot de recompensas e comprimentos dos episódios
    ax_ep = plt.subplot(rows, cols, total_plots)
    episodes = range(1, len(completed_rewards) + 1)
    ax_ep.plot(episodes, completed_rewards, label="Completed Rewards", color="black")
    ax_ep.plot(episodes, completed_lengths, label="Completed Lengths", color="gray")
    ax_ep.set_ylabel("Geral", fontsize=8)
    ax_ep.legend(fontsize=6)
    ax_ep.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_ep.tick_params(axis="x", labelbottom=False, labelsize=6)
    ax_ep.tick_params(axis="y", labelsize=6)
    mean_rewards, min_rewards, max_rewards = safe_stats(completed_rewards)
    mean_lengths, min_lengths, max_lengths = safe_stats(completed_lengths)
    ax_ep.text(
        0.5,
        -0.4,
        f"Rewards -> Média: {mean_rewards:.4f} | Min: {min_rewards:.4f} | Max: {max_rewards:.4f}\n"
        f"Lengths -> Média: {mean_lengths:.4f} | Min: {min_lengths:.4f} | Max: {max_lengths:.4f}",
        transform=ax_ep.transAxes,
        ha="center",
        fontsize=6,
    )

    plt.tight_layout()
    plt.show()


import time
from functools import wraps


def medir_tempo(ativar=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not ativar:
                return func(*args, **kwargs)
            inicio = time.time()
            resultado = func(*args, **kwargs)
            fim = time.time()
            print(f"Tempo de execução de {func.__name__}: {fim - inicio:.6f} segundos")
            return resultado

        return wrapper

    return decorator
