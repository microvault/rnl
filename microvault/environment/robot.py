from dataclasses import dataclass
from typing import Tuple

import numpy as np

from microvault.engine.collision import Collision


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
    ):
        self.fov = fov
        self.max_range = max_range
        self.num_rays = num_rays
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.collision = collision
        self.lidar_angle = np.linspace(0, self.fov, self.num_rays)

    def move_robot(
        self,
        last_position_x: float,
        last_position_y: float,
        last_theta: float,
        vl: float,
        vr: float,
    ):
        epsilon = 1e-6

        v = vl
        omega = vr

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
