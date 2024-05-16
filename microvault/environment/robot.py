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

    def calculate_velocities(self, vl, vr):
        v = (self.wheel_radius / 2) * (vl + vr)
        omega = (self.wheel_radius / self.wheel_base) * (vr - vl)
        return v, omega

    def update_position(self, dt, x, y, theta, vl, vr):
        v, omega = self.calculate_velocities(vl, vr)
        theta_new = theta + omega * dt

        # Limitando a velocidade para evitar saltos muito grandes
        max_distance = v * dt
        dx = max_distance * np.cos(theta_new)
        dy = max_distance * np.sin(theta_new)

        # Calculando a nova posição com base na velocidade limitada
        x_new = min(max(0, x + dx), self.xmax)
        y_new = min(max(0, y + dy), self.ymax)

        return x_new, y_new, theta_new

    def x_advance(self, dt: int, x: np.ndarray, vx: np.ndarray) -> None:
        """
        Advances the agent's x position based on its current velocity.

        Parameters:
        dt (int): The current time step.
        x (np.ndarray): Array of agent's x positions over time.
        vx (np.ndarray): Array of agent's x velocities over time.

        Returns:
        None
        """
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
        """
        Advances the agent's y position based on its current velocity.

        Parameters:
        dt (int): The current time step.
        y (np.ndarray): Array of agent's y positions over time.
        vy (np.ndarray): Array of agent's y velocities over time.

        Returns:
        None
        """
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

    def move(self, dt, x, y, theta, vx, vy, vl, vr):
        """
        Move o robô com base nas velocidades das rodas esquerda e direita.

        Parameters:
        vl (float): Velocidade linear da roda esquerda.
        vr (float): Velocidade linear da roda direita.

        Returns:
        None
        """
        # Calcula as velocidades linear e angular do robô
        v, omega = self.calculate_velocities(vl, vr)

        # Atualiza a posição e a orientação do robô com base nas velocidades calculadas
        x[dt + 1], y[dt + 1], theta[dt + 1] = self.update_position(
            dt, x[dt], y[dt], theta[dt], vl, vr
        )

        # print("Pos X: ", x)

        # Calcula as velocidades lineares x e y do robô
        vx[dt + 1] = v * np.cos(theta[dt + 1])
        vy[dt + 1] = v * np.sin(theta[dt + 1])
