from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.configs.rewards import RewardConfig
from rnl.environment.env import NaviEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np

def _safe_plot(ax, y, color, label):
    """Plota apenas se houver dados; caso contrário esconde o subplot."""
    if len(y) == 0:
        ax.set_visible(False)
        return
    x = range(1, len(y) + 1)
    ax.plot(x, y, color=color, label=label, linewidth=1.5)
    ax.set_ylabel(label, fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.text(
        0.5, -0.25,
        f"µ {np.mean(y):.4f} | min {np.min(y):.4f} | max {np.max(y):.4f}",
        transform=ax.transAxes, ha="center", fontsize=6,
    )


def make_vect_envs(
    num_envs: int,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    use_render: bool,
    type_reward: RewardConfig,
    mode: str = "random",
):
    task_pool = ("turn", "avoid", "long")
    rng = np.random.default_rng()

    if mode == "random":
        reps = int(np.ceil(num_envs / len(task_pool)))
        base = np.tile(task_pool, reps)[:num_envs]
        chosen_modes = rng.permutation(base)
    else:
        chosen_modes = np.full(num_envs, mode)

    def make_env(i: int):
        env_mode = chosen_modes[i]
        def _init():
            print(f"[env {i}] modo → {env_mode}")
            env = NaviEnv(
                robot_config,
                sensor_config,
                env_config,
                render_config,
                use_render=use_render,
                mode=env_mode,
                type_reward=type_reward,
            )
            env.reset(seed=13 + i)
            check_env(env)
            return env
        return _init

    venv = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    return VecMonitor(venv)


def make_vect_envs_norm(
    num_envs: int,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    use_render: bool,
    type_reward: RewardConfig,
):
    """Returns subprocess-vectorized environments with custom parameters.

    :param num_envs: Number of vectorized environments
    :type num_envs: int
    :param robot_config: Robot configuration
    :type robot_config: RobotConfig
    :param sensor_config: Sensor configuration
    :type sensor_config: SensorConfig
    :param env_config: Environment configuration
    :type env_config: EnvConfig
    :param render_config: Render configuration
    :type render_config: RenderConfig
    :param use_render: Whether to render the environment
    :type use_render: bool
    :param type_reward: Reward configuration
    :type type_reward: RewardConfig
    :return: Vectorized environment
    :rtype: VecMonitor
    """

    def make_env(i):
        def _init():
            env = NaviEnv(
                robot_config,
                sensor_config,
                env_config,
                render_config,
                use_render,
                mode=env_config.type,
                type_reward=type_reward,
            )
            env.reset(seed=13 + i)
            return env

        return _init

    venv = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    return VecMonitor(venv)
