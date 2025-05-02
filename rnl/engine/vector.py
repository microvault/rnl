from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.configs.rewards import RewardConfig
from rnl.environment.env import NaviEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np

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
    task_pool = ("turn", "avoid")
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
    """Returns async-vectorized gym environments with custom parameters.

    :param env_name: Gym environment name or custom environment class
    :type env_name: str or type
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    :param env_kwargs: Additional keyword arguments for the environment
    :type env_kwargs: dict
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

    return gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)])
