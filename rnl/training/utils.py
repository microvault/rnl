import gymnasium as gym
from rnl.environment.env import NaviEnv
from rnl.configs.config import RobotConfig, SensorConfig, EnvConfig, RenderConfig


def create_env(num_envs):
    robot_config = RobotConfig(
        base_radius=0.105,
        vel_linear=[-1.0, 1.0],
        vel_angular=[-0.5, 0.5],
        wheel_distance=0.5,
        weight=5.0,
        threshold=0.5,
        collision=0.3,
        path_model="None"
    )
    sensor_config = SensorConfig(
        fov=240.0,
        num_rays=36,
        min_range=0.1,
        max_range=5.0
    )
    env_config = EnvConfig(
        scalar=30,
        folder_map="",
        name_map="",
        timestep=1000
    )
    render_config = RenderConfig(
        controller=False,
        debug=True,
        plot=False
    )

    def make_env(i):
        def _init():
            env = NaviEnv(
                robot_config,
                sensor_config,
                env_config,
                render_config,
                False,
                mode="easy-00"
            )
            env.reset(seed=13 + i)
            return env

        return _init

    return gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)])

def make_env():
    robot_config = RobotConfig(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=0.1,  # 4 # 0.03
        collision=0.075,  # 2 # 0.075
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
        timestep=1000
    )
    render_config = RenderConfig(
        controller=False,
        debug=True,
        plot=False
    )

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        False,
        mode="easy-00"
    )

    return env
