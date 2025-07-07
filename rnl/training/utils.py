from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.configs.rewards import RewardConfig
from rnl.environment.env import NaviEnv

def make_environemnt(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    type_reward: RewardConfig,
):
    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        False,
        type_reward=type_reward,
    )

    return env
