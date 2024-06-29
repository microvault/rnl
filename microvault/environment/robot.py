from dataclasses import dataclass
from typing import Tuple

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .engine.collision import filter_segment, lidar_intersections


@dataclass
class Robot:
    def __init__(
        self,
        time: int = 100,
        min_radius: int = 1,
        max_radius: int = 3,
        max_grid: int = 15,
        wheel_radius: float = 0.3,
        wheel_base: float = 0.3,
        fov: int = 2 * np.pi,
        num_rays: int = 40,
        max_range: int = 6,
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
        self.lidar_angle = np.linspace(0, self.fov, self.num_rays)
        self.dt = 1

    def init_agent(self, ax: Axes3D) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ]:
        """
        Initializes the agent's parameters for the simulation.

        Parameters:
        ax (Axes3D): The 3D axes to be used for plotting.

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
        theta = np.zeros(self.time)
        radius = np.random.uniform(self.min_radius, self.max_radius)

        return (x, y, theta, self.lidar_angle, radius)

    def move(
        self,
        i: int,
        x: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        vl: float,
        vr: float,
    ) -> None:
        v_right = vr * self.wheel_radius
        v_left = vl * self.wheel_radius

        v = (v_right + v_left) / 2
        omega = (v_right - v_left) / self.wheel_base

        theta_prev = theta[i - 1]
        theta_new = theta_prev + omega * self.dt

        # Ensure theta stays within -2π to 2π
        if theta_new > 2 * np.pi or theta_new < -2 * np.pi:
            theta_new = 0

        theta[i] = theta_new

        x_temp = x[i - 1] + v * np.cos(theta_new) * self.dt
        y_temp = y[i - 1] + v * np.sin(theta_new) * self.dt

        x_new = max(0.0, float(min(x_temp, self.xmax)))
        y_new = max(0.0, float(min(y_temp, self.ymax)))

        x[i] = x_new
        y[i] = y_new

    def sensor(
        self, i: int, x: np.ndarray, y: np.ndarray, segments
    ) -> Tuple[np.ndarray, np.ndarray]:
        seg = filter_segment(segments, x[i], y[i], 6)

        intersections, measurements = lidar_intersections(
            x[i], y[i], self.max_range, self.lidar_angle, seg
        )

        return intersections, measurements

    def step(self):
        pass
