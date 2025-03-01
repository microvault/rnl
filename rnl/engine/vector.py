import gymnasium as gym

from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.environment.env import NaviEnv
from rnl.configs.actions import FastActions, SlowActions


def make_vect_envs(
    num_envs: int,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    use_render: bool,
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
            slow_actions = SlowActions()
            fast_actions = FastActions()
            env = NaviEnv(
                robot_config, sensor_config, env_config, render_config, use_render,
                actions_cfg=slow_actions
            )
            env.reset(seed=13 + i)
            return env
        return _init

    return gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)])
