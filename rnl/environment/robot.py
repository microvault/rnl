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
        wheel_radius: float = 0.3,
        wheel_base: float = 0.3,
        fov: float = 4 * np.pi,
        num_rays: int = 20,
        max_range: float = 6.0,
        min_range: float = 1.0,
        mass: float = 10.0,  # Massa do robô
        inertia: float = 100.0,
    ):
        self.fov = fov
        self.max_range = max_range
        self.num_rays = num_rays
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.collision = collision
        self.lidar_angle = np.linspace(0, self.fov, self.num_rays)

        self.space = pymunk.Space()

        self.body = pymunk.Body(mass, inertia)
        self.body.position = Vec2d(0, 0)

        self.shape = pymunk.Circle(self.body, self.wheel_radius)
        self.shape.friction = 0.9
        self.shape.elasticity = 0.5

        self.space.add(self.body, self.shape)

    def apply_forces(self, vl, vr):
        v = self.wheel_radius * (vl + vr) / 2
        omega = self.wheel_radius * (vr - vl) / self.wheel_base

        force = v * self.body.mass
        torque = omega * self.body.moment

        direction = Vec2d(np.cos(self.body.angle), np.sin(self.body.angle))
        self.body.apply_force_at_local_point(force * direction)
        self.body.torque += torque

    def move_robot(
        self,
        last_position_x: float,
        last_position_y: float,
        last_theta: float,
        vl: float,
        vr: float,
    ):
        epsilon = 1e-6

        # v = vl
        # omega = vr
        v = self.wheel_radius * (vl + vr) / 2
        omega = self.wheel_radius * (vr - vl) / self.wheel_base

        theta_new = last_theta + omega
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        if abs(omega) < epsilon:
            x_new = last_position_x + v * np.cos(last_theta)
            y_new = last_position_y + v * np.sin(last_theta)
        else:
            radius = v / omega if omega != 0 else float("inf")
            cx = last_position_x - radius * np.sin(last_theta)
            cy = last_position_y + radius * np.cos(last_theta)
            delta_theta = omega
            x_new = cx + radius * np.sin(last_theta + delta_theta)
            y_new = cy - radius * np.cos(last_theta + delta_theta)

        return x_new, y_new, theta_new

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
