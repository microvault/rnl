from typing import Tuple

import numpy as np


class Robot:
    def __init__(
        self,
        time: int = 10,
        min_radius: int = 1,
        max_radius: int = 3,
        xmax: int = 10,
        ymax: int = 10,
    ):
        self.time = time
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.xmax = xmax
        self.ymax = ymax

    def init_agent(self, ax) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ]:

        # dynamic variables
        x = np.zeros(self.time)  # position x
        y = np.zeros(self.time)  # position y
        sp = np.zeros(self.time)  # speed magnitude
        theta = np.zeros(self.time)  # angle of velocity
        vx = np.zeros(self.time)  # velocity x
        vy = np.zeros(self.time)  # velocity y
        radius = np.random.uniform(
            self.min_radius, self.max_radius
        )  # radius of the robot

        return (x, y, sp, theta, vx, vy, radius)

    def x_advance(self, dt: int, x: np.ndarray, vx: np.ndarray) -> None:
        if (self.time - 1) != dt:
            if x[dt] + vx[dt] >= self.xmax or x[dt] + vx[dt] <= 0:
                x[dt + 1] = x[dt] - vx[dt]
                vx[dt + 1] = -vx[dt]
            else:
                x[dt + 1] = x[dt] + vx[dt]
                vx[dt + 1] = vx[dt]
        else:
            if x[dt] + vx[dt] >= self.xmax or x[dt] + vx[dt] <= 0:
                x[dt] = x[dt] - vx[dt]
                vx[dt] = -vx[dt]
            else:
                x[dt] = x[dt] + vx[dt]
                vx[dt] = vx[dt]

    def y_advance(self, dt: int, y: np.ndarray, vy: np.ndarray) -> None:
        if (self.time - 1) != dt:
            if y[dt] + vy[dt] >= self.ymax or y[dt] + vy[dt] <= 0:
                y[dt + 1] = y[dt] - vy[dt]
                vy[dt + 1] = -vy[dt]
            else:
                y[dt + 1] = y[dt] + vy[dt]
                vy[dt + 1] = vy[dt]
        else:
            if y[dt] + vy[dt] >= self.ymax or y[dt] + vy[dt] <= 0:
                y[dt] = y[dt] - vy[dt]
                vy[dt] = -vy[dt]
            else:
                y[dt] = y[dt] + vy[dt]
                vy[dt] = vy[dt]
