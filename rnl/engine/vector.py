from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.configs.rewards import RewardConfig
from rnl.environment.env import NaviEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env

import numpy as np
import time

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
    task_pool = ("long", "turn", "avoid")
    rng = np.random.default_rng(int(time.time_ns() % 2**32))
    chosen_modes = rng.choice(task_pool, size=num_envs)

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
