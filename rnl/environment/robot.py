from dataclasses import dataclass

import numpy as np
import pymunk

from rnl.configs.config import RobotConfig


@dataclass
class Robot:
    """
    A class representing a robot with physical and sensor properties.
    """

    def __init__(self, robot_config: RobotConfig):
        """
        Initialize additional attributes after dataclass initialization.
        """
        # Constant acceleration due to gravity
        g = 9.81  # m/s²

        # Calculating the mass of the robot
        self.mass = robot_config.weight / g  # kg
        self.robot_radius = robot_config.base_radius
        self.wheel_base = robot_config.wheel_distance

        # Calculating the moment of inertia of the robot
        # I = (1/2) * m * r²
        self.inertia = 0.5 * self.mass * self.robot_radius**2
        self.body = pymunk.Body(self.mass, self.inertia)

    @staticmethod
    def create_space() -> pymunk.Space:
        """
        Create and return a new pymunk space with no gravity.
        """
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        return space

    def create_robot(
        self,
        space: pymunk.Space,
        friction: float = 0.4,
        damping: float = 0.1,
        position_x: float = 0.0,
        position_y: float = 0.0,
    ) -> pymunk.Body:
        """
        Create and add the robot to the given pymunk space.
        """
        body = pymunk.Body(self.mass, self.inertia)
        body.position = (position_x, position_y)
        shape = pymunk.Circle(body, self.robot_radius)
        shape.friction = friction
        shape.damping = damping
        space.add(body, shape)
        return body

    def move_robot(
        self,
        space: pymunk.Space,
        robot_body: pymunk.Body,
        v_linear: float,
        v_angular: float,
    ) -> None:
        """
        Move the robot in the space with given linear and angular velocities.
        """
        direction = pymunk.Vec2d(np.cos(robot_body.angle), np.sin(robot_body.angle))
        robot_body.velocity = v_linear * direction
        robot_body.angular_velocity = v_angular
        space.step(1 / 60)

    def reset_robot(self, robot_body: pymunk.Body, x: float, y: float) -> None:
        """
        Reset the robot's position and velocity.
        """
        robot_body.position = (x, y)
        robot_body.angle = 0
        robot_body.velocity = (0, 0)
        robot_body.angular_velocity = 0
