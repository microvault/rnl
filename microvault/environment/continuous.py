import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, art3d
from shapely.geometry import Point, Polygon

from .generate import Generator
from .robot import Robot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from algorithm.td3agent import Agent

# car racing
# - 8 m/s
# - 0 m/s
# - radius = 56 cm
# - peso = 2.64 kg


class Continuous:
    def __init__(
        self,
        timestep: int = 100,  # max step
        size: float = 3.0,  # size robot
        fps: int = 100,  # 10 frames per second
        random: int = 1000,  # 100 random points
        max_linear: float = 1.0,  # 0.2 m/s
        min_linear: float = 0.0,  # 0.1 m/s
        max_angular: float = 1.0,  # 0.2 rad/s
        min_angular: float = -1.0,  # 0.1 rad/s
        num_rays: int = 20,  # num range lidar
        fov: int = 2 * np.pi,  # 360 degrees
        max_range: int = 6,  # max range
        epochs: int = 1000,  # 1000 epochs
        threshold: float = 0.1,  # 0.1 threshold
        grid_lenght: int = 10,  # TODO: error < 5 -> [5 - 15]
    ):
        self.timestep = timestep
        self.max_timestep = timestep
        self.size = size
        self.fps = fps
        self.max_linear = max_linear
        self.min_linear = min_linear
        self.max_angular = max_angular
        self.min_angular = min_angular
        self.threshold = threshold

        self.grid_lenght = grid_lenght

        self.xmax = grid_lenght - 0.25
        self.ymax = grid_lenght - 0.25

        self.segments = None
        self.poly = None
        self.epoch = 0
        self.max_epoch = epochs

        # TODO: remove the team and remove in array format
        self.target_x = 0
        self.target_y = 0

        self.laser_scatters = []

        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))

        self.generator = Generator(grid_lenght=grid_lenght, random=random)
        self.robot = Robot(
            self.timestep,
            1,
            3,
            max_grid=grid_lenght,
            max_range=max_range,
            num_rays=num_rays,
            fov=fov,
        )
        # state_dim = 20 (lidar) + 2 (velocity) + 1 (dist target)
        self.agent = Agent(
            state_size=23,
            action_size=2,
            max_action=1,
            min_action=0.1,
            noise=0.2,
            noise_std=0.1,
            noise_clip=0.5,
            pretraining=False,
        )

        self.ax.remove()
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

        (
            self.x,
            self.y,
            self.theta,
            self.lidar_angle,
            self.radius,
        ) = self.robot.init_agent(self.ax)

        self.target = self.ax.plot3D(
            np.random.uniform(0, self.xmax),
            np.random.uniform(0, self.ymax),
            0,
            marker="x",
            markersize=self.size,
            color="red",
        )[0]

        self.agents = self.ax.plot3D(
            self.x,
            self.y,
            0,
            marker="o",
            markersize=self.radius,
            color="orange",
        )[0]

        self.ani = None
        self.reset_flag = False

        self._init_animation(self.ax)

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

        path, poly, seg = self.generator.world()

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
            self._get_label(0, 0),
        )

        self.label.set_fontsize(14)
        self.label.set_fontweight("normal")
        self.label.set_color("#666666")

        self.fig.subplots_adjust(left=0, right=1, bottom=0.1, top=1)

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
    def _get_label(timestep: int, epoch: int) -> str:
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
        return line1 + line2 + line3

    @staticmethod
    def distance(x: float, y: float, target_x: float, target_y: float) -> float:
        return np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

    def reset_world(self) -> None:
        """
        Resets the environment.

        Clears existing patches from the plot, generates a new map, places the target and agent randomly within the environment,
        and initializes their velocities and directions randomly.

        Returns:
        None
        """
        self.epoch += 1

        for patch in self.ax.patches:
            patch.remove()

        new_map_path, poly, seg = self.generator.world()
        self.segments = seg
        self.poly = poly

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

    def plot_anim(
        self,
        i: int,
        intersections: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        target_x: np.ndarray,
        target_y: np.ndarray,
        epoch: int,
    ) -> None:
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
            [x[i]],
            [y[i]],
            [0],
        )

        self.target.set_data_3d(
            [target_x],
            [target_y],
            [0],
        )

        self.label.set_text(self._get_label(i, epoch))

    def reset_agent(self, poly) -> None:
        self.theta[0] = np.random.uniform(0, 2 * np.pi)
        self.x[0] = np.random.uniform(0, self.xmax)
        self.y[0] = np.random.uniform(0, self.ymax)

        agent_inside = False

        while not agent_inside:
            self.x[0] = np.random.uniform(0, self.xmax)
            self.y[0] = np.random.uniform(0, self.ymax)

            if self._ray_casting(poly, self.x[0], self.y[0]):
                agent_inside = True

    def step(self, i: int) -> None:
        """
        Advances the simulation by one step.

        Parameters:
        i (int): The current time step.

        Returns:
        None
        """

        vr = np.random.uniform(self.min_angular, self.max_angular)
        vl = np.random.uniform(self.min_linear, self.max_linear)
        reward = 0
        done = False

        if i == 0:
            self.reset_agent(self.poly)

        else:
            self.robot.move(i, self.x, self.y, self.theta, vl, vr)

            intersections, measurements = self.robot.sensor(
                i, self.x, self.y, self.segments
            )
            dist = self.distance(self.x[i], self.y[i], self.target_x, self.target_y)
            state = np.concatenate(
                (
                    np.array(measurements, dtype=np.float32),
                    np.array([vr], dtype=np.float32),
                    np.array([vl], dtype=np.float32),
                    np.array([dist], dtype=np.float32),
                )
            )
            action = self.agent.predict(state)

            self.agent.step(state, action, reward, state, done)

            self.plot_anim(
                i,
                intersections,
                self.x,
                self.y,
                self.target_x,
                self.target_y,
                self.epoch,
            )

            for m in measurements:
                if m <= self.threshold:
                    self.agent.learn(self.max_timestep, self.epoch)
                    self.reset_world()
                    self.reset_agent(self.poly)
                    self.ani.frame_seq = self.ani.new_frame_seq()
                    self.timestep = self.max_timestep

    def render(self, plot: str = "local") -> None:
        """
        Displays the simulation.

        Parameters:
        plot (bool): Whether to plot the animation (default is False).

        Returns:
        None
        """

        self.ani = animation.FuncAnimation(
            self.fig,
            self.step,
            init_func=self.reset_world,
            blit=False,
            frames=self.timestep,
            interval=self.fps,
        )

        if plot == "local":
            plt.show()
        elif plot == "video":
            self.ani.save("anim.mp4", fps=self.time)
        else:
            pass

    def trainer(self, visualize: str = ""):
        self.render(visualize)


# agent = Continuous()
# agent.trainer(visualize="local")
