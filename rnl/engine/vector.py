import gymnasium as gym

from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.configs.rewards import RewardConfig
from rnl.environment.env import NaviEnv
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
    task_pool = ("long", "turn", "all", "avoid")

    def make_env(i: int):
        def _init():
            chosen_mode = (
                np.random.choice(task_pool) if mode == "random" else mode
            )
            print("mode: ", chosen_mode)
            env = NaviEnv(
                robot_config,
                sensor_config,
                env_config,
                render_config,
                use_render=use_render,
                mode=chosen_mode,
                type_reward=type_reward,
            )
            env.reset(seed=13 + i)
            return env
        return _init

    return gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)])
