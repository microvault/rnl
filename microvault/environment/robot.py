from dataclasses import dataclass
from typing import Tuple

import numpy as np

from microvault.engine.collision import Collision


@dataclass
class Robot:
    def __init__(
        self,
        collision: Collision,
        time: int = 100,
        min_radius: float = 1.0,
        max_radius: float = 3.0,
        max_grid: int = 15,
        wheel_radius: float = 0.3,
        wheel_base: float = 0.3,
        fov: float = 2 * np.pi,
        num_rays: int = 10,
        max_range: float = 6.0,
    ):
        self.time = time
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.xmax = max_grid
        self.ymax = max_grid
        self.fov = fov
        self.max_range = max_range
        self.num_rays = num_rays
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.collision = collision
        self.lidar_angle = np.linspace(0, self.fov, self.num_rays)
        self.dt = 1

    def init_agent(self) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ]:
        """
        Initializes the agent's parameters for the simulation.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            Tuple containing the following:
            - x (np.ndarray): Array of agent's x positions over time.
            - y (np.ndarray): Array of agent's y positions over time.
            - sp (np.ndarray): Array of agent's speed magnitudes over time.
            - theta (np.ndarray): Array of agent's angles of velocity over time.
            - vx (np.ndarray): Array of agent's x velocities over time.
            - vy (np.ndarray): Array of agent's y velocities over time.
            - radius (float): Radius of the robot.
        """
        x = np.zeros(self.time)
        y = np.zeros(self.time)
        vr = np.zeros(self.time)
        vl = np.zeros(self.time)
        theta = np.zeros(self.time)
        measurements = np.zeros((self.time, self.num_rays))
        radius = np.random.uniform(self.min_radius, self.max_radius)

        return (x, y, theta, self.lidar_angle, measurements, vr, vl, radius)

    def move_robot(
        self,
        x: float,
        y: float,
        theta: float,
        vl: float,
        vr: float,
    ):
        v_right = vr * self.wheel_radius
        v_left = vl * self.wheel_radius

        v = (v_right + v_left) / 2
        omega = (v_right - v_left) / self.wheel_base

        theta_new = theta + omega * self.dt

        # Ensure theta stays within -2π to 2π
        if theta_new > 2 * np.pi or theta_new < -2 * np.pi:
            theta_new = 0

        x_temp = x + v * np.cos(theta_new) * self.dt
        y_temp = y + v * np.sin(theta_new) * self.dt

        x_new = max(0.0, float(min(x_temp, self.xmax)))
        y_new = max(0.0, float(min(y_temp, self.ymax)))

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

    def step(self):
        pass
