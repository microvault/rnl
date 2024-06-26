from dataclasses import dataclass
from typing import Tuple

import numpy as np
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class Robot:
    def __init__(
        self,
        time: int = 100,
        min_radius: int = 1,
        max_radius: int = 3,
        xmax: int = 15,
        ymax: int = 15,
        wheel_radius: float = 0.3,
        wheel_base: float = 0.3,
    ):
        self.time = time
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.xmax = xmax
        self.ymax = ymax

        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.body = None

    def init_agent(self, ax: Axes3D) -> Tuple[
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
        sp = np.zeros(self.time)
        theta = np.zeros(self.time)
        vx = np.zeros(self.time)
        vy = np.zeros(self.time)
        radius = np.random.uniform(self.min_radius, self.max_radius)

        return (x, y, sp, theta, vx, vy, radius)

    def move(self, dt, x, y, theta, vl, vr) -> None:
        delta_t = 1  # Assuming time step of 1 unit

        v_right = vr * self.wheel_radius
        v_left = vl * self.wheel_radius

        v = (v_right + v_left) / 2
        omega = (v_right - v_left) / self.wheel_base

        theta_prev = theta[dt - 1]
        theta_new = theta_prev + omega * delta_t

        # Ensure theta stays within -2π to 2π
        if theta_new > 2 * np.pi or theta_new < -2 * np.pi:
            theta_new = 0

        theta[dt] = theta_new

        x_temp = x[dt - 1] + v * np.cos(theta_new) * delta_t
        y_temp = y[dt - 1] + v * np.sin(theta_new) * delta_t

        x_new = max(0.0, float(min(x_temp, self.xmax)))
        y_new = max(0.0, float(min(y_temp, self.ymax)))

        x[dt] = x_new
        y[dt] = y_new
