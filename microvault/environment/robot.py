from collections import namedtuple
from typing import Tuple

import numpy as np
from matplotlib.patches import Circle


class Robot:
    def __init__(
        self,
        num_agents,
        time,
        min_radius=1,
        max_radius=3,
        xmax=10,
        ymax=10,
    ):

        self.agent = namedtuple(
            "Agent",
            field_names=["x", "y", "theta", "radius", "vx", "vy"],
        )

        self.num_agents = num_agents
        self.time = time
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.xmax = xmax
        self.ymax = ymax

    def init_agent(
        self, ax
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list
    ]:

        # dynamic variables
        x = np.zeros((self.num_agents, self.time))  # position x
        y = np.zeros((self.num_agents, self.time))  # position y
        sp = np.zeros((self.num_agents, self.time))  # speed magnitude
        theta = np.zeros((self.num_agents, self.time))  # angle of velocity
        vx = np.zeros((self.num_agents, self.time))  # velocity x
        vy = np.zeros((self.num_agents, self.time))  # velocity y
        radius = np.random.uniform(
            self.min_radius, self.max_radius, self.num_agents
        )  # radius of the robot

        agents = []

        for a in range(self.num_agents):
            position_x = np.random.uniform(0, self.xmax)
            position_y = np.random.uniform(0, self.ymax)
            agent = ax.plot3D(
                position_x, position_y, 0, marker="o", markersize=radius[a]
            )[0]
            agents.append(agent)

        return (x, y, sp, theta, vx, vy, agents)

    def x_advance(self, agents, i, x, vx) -> None:
        for a in range(0, self.num_agents):
            if (self.time - 1) != i:
                if x[a, i] + vx[a, i] >= self.xmax or x[a, i] + vx[a, i] <= 0:
                    x[a, i + 1] = x[a, i] - vx[a, i]
                    vx[a, i + 1] = -vx[a, i]
                else:
                    x[a, i + 1] = x[a, i] + vx[a, i]
                    vx[a, i + 1] = vx[a, i]
            else:
                if x[a, i] + vx[a, i] >= self.xmax or x[a, i] + vx[a, i] <= 0:
                    x[a, i] = x[a, i] - vx[a, i]
                    vx[a, i] = -vx[a, i]
                else:
                    x[a, i] = x[a, i] + vx[a, i]
                    vx[a, i] = vx[a, i]

    def y_advance(self, agents, i, y, vy) -> None:
        for a in range(0, self.num_agents):
            if (self.time - 1) != i:
                if y[a, i] + vy[a, i] >= self.ymax or y[a, i] + vy[a, i] <= 0:
                    y[a, i + 1] = y[a, i] - vy[a, i]
                    vy[a, i + 1] = -vy[a, i]
                else:
                    y[a, i + 1] = y[a, i] + vy[a, i]
                    vy[a, i + 1] = vy[a, i]
            else:
                if y[a, i] + vy[a, i] >= self.ymax or y[a, i] + vy[a, i] <= 0:
                    y[a, i] = y[a, i] - vy[a, i]
                    vy[a, i] = -vy[a, i]
                else:
                    y[a, i] = y[a, i] + vy[a, i]
                    vy[a, i] = vy[a, i]

    def overlaps(self, x, y, other_x, other_y, other_radius, radius):
        """Does the circle of this Robot overlap that of other?"""

        return np.hypot(other_x - x, other_y - y) < radius + other_radius
