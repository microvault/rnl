import numpy as np
import pytest

from microvault.environment.robot import Robot


@pytest.fixture
def robot_instance():
    return Robot()


def test_init_robot(robot_instance):

    num_agents = 2
    time = 10
    x, y, sp, theta, vx, vy = robot_instance.init_agent(num_agents, time)
    assert x.shape == (num_agents, time)
    assert y.shape == (num_agents, time)
    assert sp.shape == (num_agents, time)
    assert theta.shape == (num_agents, time)
    assert vx.shape == (num_agents, time)
    assert vy.shape == (num_agents, time)

    assert np.all(x == 0)
    assert np.all(y == 0)
    assert np.all(sp == 0)
    assert np.all(theta == 0)
    assert np.all(vx == 0)
    assert np.all(vy == 0)


def test_x_direction(robot_instance):
    robot_instance.x_direction(1, 1, 2, 10, np.zeros((2, 10)), np.zeros((2, 10)), 10)
    assert True


def test_y_direction(robot_instance):
    robot_instance.y_direction(1, 1, 2, 10, np.zeros((2, 10)), np.zeros((2, 10)), 10)
    assert True
