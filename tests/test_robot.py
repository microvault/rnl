import numpy as np
import pymunk
import pytest

from rnl.configs.config import RobotConfig, SensorConfig
from rnl.environment.robot import Robot
from rnl.environment.sensor import SensorRobot


@pytest.fixture
def robot():
    robot_config = RobotConfig()
    return Robot(robot_config)


def test_create_space():
    space = Robot.create_space()
    assert isinstance(space, pymunk.Space)
    assert space.gravity == (0.0, 0.0)


def test_create_robot(robot):
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    position_x = 2.0
    position_y = 3.0
    robot_body = robot.create_robot(space, position_x=position_x, position_y=position_y)

    assert isinstance(robot_body, pymunk.Body)
    assert robot_body.position.x == pytest.approx(position_x)
    assert robot_body.position.y == pytest.approx(position_y)
    assert robot_body.velocity == (0.0, 0.0)
    assert robot_body.angular_velocity == 0.0

    shapes = space.shapes
    assert len(shapes) == 1
    shape = shapes[0]
    assert isinstance(shape, pymunk.Circle)
    assert shape.body == robot_body
    assert shape.radius == pytest.approx(robot.robot_radius)
    assert shape.friction == 0.9


def test_move_robot(robot):
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    robot_body = robot.create_robot(space)
    v_linear = 1.0
    v_angular = 0.5

    robot.move_robot(space, robot_body, v_linear, v_angular)


def test_reset_robot(robot):
    space = pymunk.Space()
    robot_body = robot.create_robot(space, position_x=5.0, position_y=5.0)
    robot_body.angle = np.pi / 4
    robot_body.velocity = (2.0, 3.0)
    robot_body.angular_velocity = 1.0

    robot.reset_robot(robot_body, x=1.0, y=1.0, angle=0.0)

    assert robot_body.position.x == pytest.approx(1.0)
    assert robot_body.position.y == pytest.approx(1.0)
    assert robot_body.angle == pytest.approx(0.0)
    assert robot_body.velocity == (0.0, 0.0)
    assert robot_body.angular_velocity == pytest.approx(0.0)


def test_sensor_robot():
    sensor_config = SensorConfig()
    robot = SensorRobot(sensor_config)

    x, y, theta = 1.0, 1.0, 0.0
    segments = [
        (0.0, 6.0, 1.0, 6.0),
        (1.0, 6.0, 2.0, 6.0),
        (2.0, 6.0, 3.0, 6.0),
        (3.0, 6.0, 4.0, 6.0),
        (4.0, 6.0, 5.0, 6.0),
        (5.0, 6.0, 6.0, 6.0),
        (6.0, 6.0, 6.0, 5.0),
        (6.0, 5.0, 6.0, 4.0),
        (6.0, 4.0, 6.0, 3.0),
        (6.0, 3.0, 6.0, 2.0),
        (6.0, 2.0, 6.0, 1.0),
        (6.0, 1.0, 6.0, 0.0),
        (6.0, 0.0, 5.0, 0.0),
        (5.0, 0.0, 4.0, 0.0),
        (4.0, 0.0, 3.0, 0.0),
        (3.0, 0.0, 2.0, 0.0),
        (2.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 2.0, 0.0, 3.0),
        (0.0, 3.0, 0.0, 4.0),
        (0.0, 4.0, 0.0, 5.0),
        (0.0, 5.0, 0.0, 6.0),
        (0.0, 6.0, 0.0, 6.0),
        (1.5, 2.0, 2.0, 1.5),
        (2.0, 1.5, 2.5, 1.0),
        (2.5, 1.0, 2.0, 0.5),
        (2.0, 0.5, 1.5, 1.0),
        (1.5, 1.0, 1.0, 1.5),
        (1.0, 1.5, 0.5, 2.0),
        (0.5, 2.0, 1.0, 2.5),
        (1.0, 2.5, 1.5, 2.0),
        (1.5, 2.0, 1.5, 2.0),
        (5.5, 1.0, 5.0, 0.5),
        (5.0, 0.5, 4.5, 1.0),
        (4.5, 1.0, 5.0, 1.5),
        (5.0, 1.5, 5.5, 1.0),
        (5.5, 1.0, 5.5, 1.0),
        (3.5, 3.0, 3.0, 2.5),
        (3.0, 2.5, 2.5, 3.0),
        (2.5, 3.0, 3.0, 3.5),
        (3.0, 3.5, 3.5, 3.0),
        (3.5, 3.0, 3.5, 3.0),
        (1.5, 4.0, 1.0, 3.5),
        (1.0, 3.5, 0.5, 4.0),
        (0.5, 4.0, 1.0, 4.5),
        (1.0, 4.5, 1.5, 4.0),
        (1.5, 4.0, 1.5, 4.0),
        (4.5, 5.0, 5.0, 4.5),
        (5.0, 4.5, 5.5, 4.0),
        (5.5, 4.0, 5.0, 3.5),
        (5.0, 3.5, 4.5, 4.0),
        (4.5, 4.0, 4.0, 4.5),
        (4.0, 4.5, 3.0, 4.5),
        (3.0, 4.5, 2.5, 5.0),
        (2.5, 5.0, 3.0, 5.5),
        (3.0, 5.5, 4.0, 5.5),
        (4.0, 5.5, 4.5, 5.0),
        (4.5, 5.0, 4.5, 5.0),
    ]

    intersections, measurements = robot.sensor(x, y, theta, segments)
