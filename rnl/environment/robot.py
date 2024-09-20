from dataclasses import dataclass
from typing import Tuple

import numpy as np

from rnl.engine.collision import Collision

import pymunk
from pymunk import Vec2d


@dataclass
class Robot:
    def __init__(
        self,
        collision: Collision,
        robot_radius: float = 0.033,
        wheel_base: float = 0.16,
        fov: float = 4 * np.pi,
        num_rays: int = 20,
        max_range: float = 6.0,
        min_range: float = 1.0,
        mass: float = 1.0,
        inertia: float = 0.3,
    ):

        self.fov = fov
        self.max_range = max_range
        self.num_rays = num_rays
        self.robot_radius = robot_radius
        self.wheel_base = wheel_base
        self.collision = collision
        self.inertia = inertia
        self.mass = mass
        self.lidar_angle = np.linspace(0, self.fov, self.num_rays)

        self.body = pymunk.Body(mass, inertia)

    @staticmethod
    def create_space():
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)

        return space

    def create_robot(self, space):
        body = pymunk.Body(self.mass, self.inertia)
        body.position = (0, 0)

        shape = pymunk.Circle(body, self.robot_radius)
        shape.friction = 0.4
        shape.damping = 0.1

        space.add(body, shape)

        return body

    def move_robot(self, space, robot_body, v_linear, v_angular):
        angle = robot_body.angle

        direction = Vec2d(np.cos(robot_body.angle), np.sin(robot_body.angle))
        robot_body.velocity = v_linear * direction
        robot_body.angular_velocity = v_angular

        space.step(1 / 60)

    def reset_robot(self, robot_body, x, y):
        robot_body.position = (x, y)
        robot_body.angle = 0
        robot_body.velocity = (0, 0)
        robot_body.angular_velocity = 0

    def sensor(
        self, x: float, y: float, segments: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        seg = self.collision.filter_segments(segments, x, y, 6)

        intersections = self.collision.lidar_intersection(
            x, y, self.max_range, self.lidar_angle, seg
        )

        measurements = self.collision.lidar_measurement(
            x, y, self.max_range, self.lidar_angle, seg
        )

        return intersections, measurements
