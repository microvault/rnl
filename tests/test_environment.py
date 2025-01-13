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


def test_spawn_robot_and_goal_times(sample_polygon):
    """
    Testa a função spawn_robot_and_goal chamando 1000 vezes,
    verificando se robô e objetivo estão dentro do polígono
    e atendem às distâncias exigidas.
    """
    for i in range(1000):
        print(f"Teste {i}")
        robo_pos, goal_pos = spawn_robot_and_goal(
            sample_polygon,
            robot_clearance=1.0,
            goal_clearance=1.0,
            min_robot_goal_dist=2.0,
            max_tries=1000,
        )
        robo_point = Point(robo_pos)
        goal_point = Point(goal_pos)

        safe_poly_robot = sample_polygon.buffer(-1.0)
        safe_poly_goal = sample_polygon.buffer(-1.0)
        assert safe_poly_robot.contains(
            robo_point
        ), f"Teste {i}: Robô fora da área segura"
        assert safe_poly_goal.contains(
            goal_point
        ), f"Teste {i}: Objetivo fora da área segura"

        dist = robo_point.distance(goal_point)
        assert (
            dist >= 2.0
        ), f"Teste {i}: Robô e objetivo muito próximos (distância: {dist})"
