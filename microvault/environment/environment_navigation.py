import logging
from typing import Tuple

import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from mpl_toolkits.mplot3d import Axes3D, art3d
from omegaconf import OmegaConf
from shapely.geometry import Point, Polygon

from microvault.algorithms.agent import Agent
from microvault.environment.generate_world import Generator
from microvault.environment.robot import Robot

# car racing reference
# - 8 m/s
# - 0 m/s
# - radius = 56 cm
# - peso = 2.64 kg


class NaviEnv(gym.Env):
    def __init__(
        self,
        robot=Robot,
        generator=Generator,
        agent=Agent,
        timestep: int = 100,  # max step
        size: float = 3.0,  # size robot
        random: int = 1000,  # 100 random points
        num_rays: int = 10,  # num range lidar
        fov: float = 2 * np.pi,  # 360 degrees
        max_range: float = 6.0,  # max range
        threshold: float = 0.1,  # 0.1 threshold
        grid_lenght: int = 10,  # TODO: error < 5 -> [5 - 15]
        rgb_array: bool = False,
        fps: int = 100,  # 10 frames per second
        max_episode: int = 10,
    ):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        self.generator = generator
        self.robot = robot
        self.agent = agent

        self.rgb_array = rgb_array
        self.max_epochs = max_episode
        self.timestep = timestep
        self.max_timestep = timestep
        self.size = size
        self.fps = fps
        self.threshold = threshold
        self.grid_lenght = grid_lenght
        self.xmax = grid_lenght - 0.25
        self.ymax = grid_lenght - 0.25

        self.segments = None
        self.poly = None

        # TODO: remove the team and remove in array format
        self.target_x = 0
        self.target_y = 0
        self.last_position_x = 0
        self.last_position_y = 0
        self.last_theta = 0
        self.epoch = 0

        self.init_position_x = 0
        self.init_position_y = 0
        self.init_theta = 0

        self.radius = 1.0
        self.lidar_angle = np.linspace(0, 2 * np.pi, 10)
        self.measurement = np.zeros(10)

        if rgb_array:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
            self.ax.remove()
            self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

            self.target = self.ax.plot3D(
                np.random.uniform(0, self.xmax),
                np.random.uniform(0, self.ymax),
                0,
                marker="x",
                markersize=self.size,
                color="red",
            )[0]

            self.agents = self.ax.plot3D(
                np.random.uniform(0, self.xmax),
                np.random.uniform(0, self.ymax),
                0,
                marker="o",
                markersize=self.radius,
                color="orange",
            )[0]

            self.ani = None
            self._init_animation(self.ax)

    @staticmethod
    def _ray_casting(poly: Polygon, x: float, y: float) -> bool:
        """
        Checks if a point (x, y) is inside a polygon using the ray casting algorithm.

        Parameters:
        poly (Polygon): The polygon to check the point inclusion in.
        x (float): The x-coordinate of the point to be checked.
        y (float): The y-coordinate of the point to be checked.

        Returns:
        bool: True if the point is inside the polygon, False otherwise.
        """
        center = Point(x, y)
        circle = center.buffer(0.5)
        return circle.within(poly)

    @staticmethod
    def distance(x: float, y: float, target_x: float, target_y: float) -> float:
        return np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

    def _animation(self, i):

        if i == 0:
            self.last_theta = self.init_theta
            self.last_position_x = self.init_position_x
            self.last_position_y = self.init_position_y
        else:
            vl = np.random.uniform(-1, 1)
            vr = np.random.uniform(0, 1)

            reward = np.float64(np.random.uniform(0, 10))
            done = np.bool_(np.random.choice([True, False], p=[0.01, 0.99]))

            x, y, theta = self.robot.move_robot(
                self.last_position_x, self.last_position_y, self.last_theta, vl, vr
            )

            intersections, measurement = self.robot.sensor(x, y, self.segments)
            dist = self.distance(x, y, self.target_x, self.target_y)
            states = np.concatenate(
                (
                    np.array(measurement, dtype=np.float32),
                    np.array([vr], dtype=np.float32),
                    np.array([vl], dtype=np.float32),
                    np.array([dist], dtype=np.float32),
                )
            )

            self._plot_anim(
                i,
                intersections,
                x,
                y,
                self.target_x,
                self.target_y,
                self.epoch,
            )

            self.last_theta = theta
            self.last_position_x = x
            self.last_position_y = y

            if done == True:
                self.close()

            elif i == self.max_timestep:
                self.close()

            elif self.epoch == self.max_epochs:
                self.ani.event_source.stop()
                plt.close(self.fig)

            for m in measurement:
                if m <= self.threshold:
                    self.close()

    def step(self, action):

        vl = np.random.uniform(-1, 1)
        vr = np.random.uniform(0, 1)

        reward = np.float64(np.random.uniform(0, 10))
        done = np.bool_(np.random.choice([True, False], p=[0.01, 0.99]))

        x, y, theta = self.robot.move_robot(
            self.last_position_x, self.last_position_y, self.last_theta, vl, vr
        )

        intersections, measurement = self.robot.sensor(x, y, self.segments)
        dist = self.distance(x, y, self.target_x, self.target_y)
        states = np.concatenate(
            (
                np.array(measurement, dtype=np.float32),
                np.array([vr], dtype=np.float32),
                np.array([vl], dtype=np.float32),
                np.array([dist], dtype=np.float32),
            )
        )
        self.last_theta = theta
        self.last_position_x = x
        self.last_position_y = y

        for m in self.measurement:
            if m <= self.threshold:
                return states, -10, np.bool_(False), {}

        if self.done == True:
            return states, -10, np.bool_(False), {}

        else:
            return states, reward, np.bool_(False), {}

    def reset(self):
        new_map_path, poly, seg = self.generator.world()
        self.segments = seg
        self.poly = poly

        if self.rgb_array:
            self.epoch += 1
            for patch in self.ax.patches:
                patch.remove()

            self.ax.add_patch(new_map_path)
            art3d.pathpatch_2d_to_3d(new_map_path, z=0, zdir="z")

        self.target_x = np.random.uniform(0, self.xmax)
        self.target_y = np.random.uniform(0, self.ymax)

        target_inside = False

        while not target_inside:
            self.target_x = np.random.uniform(0, self.xmax)
            self.target_y = np.random.uniform(0, self.ymax)

            if self._ray_casting(poly, self.target_x, self.target_y):
                target_inside = True

        self.init_position_x = np.random.uniform(0, self.xmax)
        self.init_position_y = np.random.uniform(0, self.ymax)
        self.init_theta = np.random.uniform(0, 2 * np.pi)

        agent_inside = False

        while not agent_inside:
            self.init_position_x = np.random.uniform(0, self.xmax)
            self.init_position_y = np.random.uniform(0, self.ymax)

            if self._ray_casting(poly, self.init_position_x, self.init_position_y):
                agent_inside = True

        intersections, measurement = self.robot.sensor(
            self.init_position_x, self.init_position_y, self.segments
        )
        dist = self.distance(
            self.init_position_x, self.init_position_y, self.target_x, self.target_y
        )

        states = np.concatenate(
            (
                np.array(measurement, dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([0], dtype=np.float32),
                np.array([dist], dtype=np.float32),
            )
        )

        info = {}
        return states, info

    def render(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self._animation,
            init_func=self.reset,
            blit=False,
            frames=self.timestep,
            interval=self.fps,
            repeat=False,
        )

        plt.show()

    def close(self):
        self.reset()
        self.ani.frame_seq = self.ani.new_frame_seq()
        self.timestep = self.max_timestep

    def _init_animation(self, ax: Axes3D) -> None:
        """
        Initializes the 3D animation by setting up the environment and camera parameters.

        Parameters:
        ax (Axes3D): The 3D axes to be used for plotting.

        Returns:
        None
        """
        ax.set_xlim(0, self.grid_lenght)
        ax.set_ylim(0, self.grid_lenght)

        # ------ Create wordld ------ #

        path, _, _ = self.generator.world()

        ax.add_patch(path)

        art3d.pathpatch_2d_to_3d(path, z=0, zdir="z")

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Hide axes
        ax.set_axis_off()

        # Set camera
        ax.elev = 20
        ax.azim = -155
        ax.dist = 1

        self.label = self.ax.text(
            0,
            0,
            0.05,
            self._get_label(0, 0, False),
        )

        self.label.set_fontsize(14)
        self.label.set_fontweight("normal")
        self.label.set_color("#666666")

        self.fig.subplots_adjust(left=0, right=1, bottom=0.1, top=1)

    @staticmethod
    def _get_label(timestep: int, epoch: int, train: bool) -> str:
        """
        Generates a label for the environment.

        Parameters:
        timestep (int): The current time step.

        Returns:
        str: The generated label containing information about the environment and the current time step.
        """
        line1 = "Environment\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        line3 = "Epochs:".ljust(14) + f"{epoch:4.0f}\n"
        line4 = "Training".ljust(14) + f"{train}\n"

        return line1 + line2 + line3 + line4

    def _plot_anim(
        self,
        i: int,
        intersections: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        target_x: np.ndarray,
        target_y: np.ndarray,
        epoch: int,
    ) -> None:

        self.label.set_text(self._get_label(i, epoch, False))

        if hasattr(self, "laser_scatters"):
            for scatter in self.laser_scatters:
                scatter.remove()
            del self.laser_scatters

        self.laser_scatters = []
        for angle, intersection in zip(self.lidar_angle, intersections):
            if intersection is not None:
                scatter = plt.scatter(
                    intersection[0], intersection[1], color="g", s=0.5
                )
                self.laser_scatters.append(scatter)

        self.agents.set_data_3d(
            [x],
            [y],
            [0],
        )

        self.target.set_data_3d(
            [target_x],
            [target_y],
            [0],
        )
