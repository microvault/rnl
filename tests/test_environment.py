import numpy as np
import pytest

from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.environment.environment_navigation import NaviEnv


@pytest.fixture
def env():
    robot_config = RobotConfig()
    sensor_config = SensorConfig()
    env_config = EnvConfig()
    render_config = RenderConfig()

    return NaviEnv(
        robot_config=robot_config,
        sensor_config=sensor_config,
        env_config=env_config,
        render_config=render_config,
        pretrained_model=False,
    )


def test_reset(env):
    state, info = env.reset()
    assert isinstance(state, np.ndarray)


def test_step(env):
    action = 0
    state, reward, done, truncated, info = env.step(action)
    assert isinstance(state, np.ndarray)
