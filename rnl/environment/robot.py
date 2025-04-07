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
        self.moment_of_inertia = 0.5 * self.mass * (self.robot_radius**2)

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
        position_x: float = 0.0,
        position_y: float = 0.0,
    ) -> pymunk.Body:
        """
        Create and add the robot to the given pymunk space.
        """
        body = pymunk.Body(self.mass, self.moment_of_inertia)
        # body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (position_x, position_y)
        # body.damping = 0.9
        # body.angular_damping = 0.9

        shape = pymunk.Circle(body, self.robot_radius)
        # shape.friction = 1.0
        shape.friction = 0.0
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
        Moves the robot using forces and torque, without setting the speed directly.
        """
        direction = pymunk.Vec2d(np.cos(robot_body.angle), np.sin(robot_body.angle))
        desired_velocity = v_linear * direction

        velocity_diff = desired_velocity - robot_body.velocity
        delta_time = 1 / 60

        force = (velocity_diff * self.mass) / delta_time
        robot_body.apply_force_at_world_point(force, robot_body.position)

        desired_angular_velocity = v_angular
        angular_velocity_diff = desired_angular_velocity - robot_body.angular_velocity
        angular_acceleration = angular_velocity_diff / delta_time
        torque = self.moment_of_inertia * angular_acceleration
        robot_body.torque += torque

    def reset_robot(
        self, robot_body: pymunk.Body, x: float, y: float, angle: float
    ) -> None:
        """
        Reset the robot's position and velocity.
        """
        robot_body.position = (x, y)
        robot_body.angle = angle
        robot_body.velocity = (0, 0)
        robot_body.angular_velocity = 0
