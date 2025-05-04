from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.configs.rewards import RewardConfig
from rnl.environment.env import NaviEnv

def make_environemnt(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    mode: str,
    type_reward: RewardConfig,
):
    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        False,
        mode=mode,
        type_reward=type_reward,
    )

    return env


def create_single_env(i):
    robot_config = RobotConfig(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=0.1,  # 4 # 0.03
        collision=0.075,  # 2 # 0.075
        noise=False,
        path_model="None",
    )
    sensor_config = SensorConfig(
        fov=270,
        num_rays=5,  # min 5 max 20
        min_range=0.0,
        max_range=3.5,  # 3.5
    )
    env_config = EnvConfig(
        scalar=100,
        folder_map="",
        name_map="",
        timestep=1000,
        obstacle_percentage=40.0,
        map_size=5,
    )
    render_config = RenderConfig(controller=False, debug=True, plot=False)

    type_reward = RewardConfig(
        params={
            "scale_orientation": 0.02,
            "scale_distance": 0.06,
            "scale_time": 0.01,
            "scale_obstacle": 0.001,
            "scale_angular": 0.005,
        },
    )

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        False,
        mode="visualize",
        type_reward=type_reward,
    )
    env.reset(seed=13 + i)
    return env
