from dataclasses import dataclass

import pymunk
import math
from rnl.configs.config import RobotConfig


@dataclass
class Robot:
    """
    A class representing a robot with physical and sensor properties.
    """

    dt: float = 1 / 60
    inv_dt: float = 60  # 1/dt

    def __init__(self, robot_config: RobotConfig) -> None:
        g = 9.81  # m/s²

        self.mass = robot_config.weight / g                     # kg
        self.radius = robot_config.base_radius                  # m
        self.wheel_base = robot_config.wheel_distance           # m

        # momento calculado em C (pymunk.moment_for_circle) → mais rápido
        self.moment_of_inertia = pymunk.moment_for_circle(
            self.mass, 0, self.radius
        )

    @staticmethod
    def create_space() -> pymunk.Space:
        """
        Create and return a new pymunk space with no gravity.
        """
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        return space

    def create_robot(
            self, space: pymunk.Space, position_x: float = 0.0, position_y: float = 0.0
        ) -> pymunk.Body:
        body = pymunk.Body(self.mass, self.moment_of_inertia)
        body.position = (position_x, position_y)

        shape = pymunk.Circle(body, self.radius)
        shape.friction = 0.0
        space.add(body, shape)
        return body

    def move_robot(
        self,
        space: pymunk.Space,
        body: pymunk.Body,
        v_linear: float,
        v_angular: float,
    ) -> None:
        dir_x, dir_y = math.cos(body.angle), math.sin(body.angle)
        desired_vx, desired_vy = v_linear * dir_x, v_linear * dir_y
        dvx, dvy = desired_vx - body.velocity.x, desired_vy - body.velocity.y

        impulse = (dvx * self.mass, dvy * self.mass)          # Δv * m
        body.apply_impulse_at_world_point(impulse, body.position)

        desired_w = v_angular
        dw = desired_w - body.angular_velocity
        torque = self.moment_of_inertia * dw * self.inv_dt     # τ = I * α
        body.torque += torque

    def reset_robot(
            self, robot_body: pymunk.Body, x: float, y: float, angle: float
        ) -> None:
        robot_body.position = (x, y)
        robot_body.angle = angle
        robot_body.velocity = (0.0, 0.0)
        robot_body.angular_velocity = 0.0
