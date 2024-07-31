import warnings

import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from mpl_toolkits.mplot3d import Axes3D, art3d

from microvault.algorithms.agent import Agent
from microvault.engine.collision import Collision
from microvault.engine.utils import (
    angle_to_goal,
    distance_to_goal,
    get_reward,
    min_laser,
)
from microvault.environment.generate_world import Generator
from microvault.environment.robot import Robot

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np.bool8.*")

# car racing reference
# - 8 m/s
# - 0 m/s
# - radius = 56 cm
# - peso = 2.64 kg


class NaviEnv(gym.Env):
    def __init__(
        self,
        robot: Robot,
        generator: Generator,
        agent: Agent,
        collision: Collision,
        timestep: int = 100,  # max step
        size: float = 3.0,  # size robot
        random: int = 1000,  # 100 random points
        num_rays: int = 10,  # num range lidar
        fov: float = 2 * np.pi,  # 360 degrees
        max_range: float = 6.0,  # max range
        threshold: float = 0.05,  # 0.1 threshold
        grid_lenght: int = 10,  # TODO: error < 5 -> [5 - 15]
        rgb_array: bool = False,
        fps: int = 100,  # 10 frames per second
        max_episode: int = 10,
        state_size: int = 14,
        controller: bool = False,
    ):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )

        self.generator = generator
        self.robot = robot
        self.agent = agent
        self.collision = collision
        self.states = None

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

        self.segments = []

        self.target_x = 0
        self.target_y = 0
        self.last_position_x = 0
        self.last_position_y = 0
        self.last_theta = 0
        self.epoch = 0

        self.radius = 1.0
        self.lidar_angle = np.linspace(0, 2 * np.pi, 10)
        self.measurement = np.zeros(10)

        if rgb_array:
            self.speed = 1.0
            self.angular_speed = 0.5

            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
            self.ax.remove()
            self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

            self.target = self.ax.plot3D(
                np.random.uniform(0, self.xmax),
                np.random.uniform(0, self.ymax),
                0,
                marker="x",
                markersize=3.0,
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

            self.ani = animation.FuncAnimation

            self._init_animation(self.ax)
            if controller:
                self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_key_press(self, event):
        if event.key == "up":
            self.speed += 0.1
        elif event.key == "down":
            self.speed -= 0.1
        elif event.key == "left":
            self.angular_speed -= 0.1
        elif event.key == "right":
            self.angular_speed += 0.1
        print(f"Speed: {self.speed}, Angular Speed: {self.angular_speed}")

    def _animation(self, i):
        actions = self.agent.predict(self.states)
        action = [(actions[0] + 1) / 2, actions[1]]

        x, y, theta = self.robot.move_robot(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            action[0],
            action[1],
        )

        intersections, measurement = self.robot.sensor(x=x, y=y, segments=self.segments)

        self._plot_anim(
            i,
            intersections,
            x,
            y,
            self.target_x,
            self.target_y,
            self.epoch,
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y)
        angle = angle_to_goal(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            self.target_x,
            self.target_y,
        )

        self.states = np.concatenate(
            (
                np.array(measurement, dtype=np.float32),
                np.array([action[0]], dtype=np.float32),
                np.array([action[1]], dtype=np.float32),
                np.array([dist], dtype=np.float32),
                np.array([angle], dtype=np.float32),
            )
        )

        self.last_theta = theta
        self.last_position_x = x
        self.last_position_y = y

        if self.epoch == self.max_epochs:
            self.ani.event_source.stop()

        collision, laser = min_laser(measurement, self.threshold)
        reward, done = get_reward(dist, action, measurement, collision)

        print(
            "\rReward {:.2f}\tMin Laser: {:.2f}\tDistance: {:.2f}\tAngle: {:.2f}\tAction: {}".format(
                reward, laser, dist, angle, str(actions)
            ),
        )

        if done:
            self.close()

    def step(self, action):

        x, y, theta = self.robot.move_robot(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            action[0],
            action[1],
        )

        intersections, measurement = self.robot.sensor(x, y, self.segments)

        dist = distance_to_goal(
            self.last_position_x, self.last_position_y, self.target_x, self.target_y
        )

        angle = angle_to_goal(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            self.target_x,
            self.target_y,
        )

        self.states = np.concatenate(
            (
                np.array(measurement, dtype=np.float32),
                np.array([action[0]], dtype=np.float32),
                np.array([action[1]], dtype=np.float32),
                np.array([dist], dtype=np.float32),
                np.array([angle], dtype=np.float32),
            )
        )
        self.last_theta = theta
        self.last_position_x = x
        self.last_position_y = y

        collision, laser = min_laser(measurement, self.threshold)
        reward, done = get_reward(dist, action, measurement, collision)

        if done:
            return self.states, reward, done, {}
        else:
            return self.states, reward, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        new_map_path, exterior, interior, all_seg = self.generator.world()
        self.segments = all_seg

        if self.rgb_array:
            self.epoch += 1
            for patch in self.ax.patches:
                patch.remove()

            self.ax.add_patch(new_map_path)
            art3d.pathpatch_2d_to_3d(new_map_path, z=0, zdir="z")

        while True:
            self.target_x = np.random.uniform(0, self.xmax)
            self.target_y = np.random.uniform(0, self.ymax)
            self.last_position_x = np.random.uniform(0, self.xmax)
            self.last_position_y = np.random.uniform(0, self.ymax)
            self.last_theta = np.random.uniform(0, 2 * np.pi)

            if self.collision.check_collision(
                exterior, interior, self.last_position_x, self.last_position_y
            ) and self.collision.check_collision(
                exterior, interior, self.target_x, self.target_y
            ):
                break

        intersections, measurement = self.robot.sensor(
            x=self.last_position_x, y=self.last_position_y, segments=all_seg
        )

        if self.rgb_array:
            self._plot_anim(
                0,
                intersections,
                self.last_position_x,
                self.last_position_y,
                self.target_x,
                self.target_y,
                self.epoch,
            )

        dist = distance_to_goal(
            self.last_position_x, self.last_position_y, self.target_x, self.target_y
        )

        angle = angle_to_goal(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            self.target_x,
            self.target_y,
        )

        self.states = np.concatenate(
            (
                np.array(measurement, dtype=np.float32),  # measurement
                np.array([0], dtype=np.float32),  # vr
                np.array([0], dtype=np.float32),  # vl
                np.array([dist], dtype=np.float32),  # distance
                np.array([angle], dtype=np.float32),  # angle
            )
        )

        info = {}
        return self.states, info

    def render(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self._animation,
            init_func=self.reset,
            blit=False,
            frames=self.timestep,
            interval=self.fps,
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

        path, _, _, _ = self.generator.world()

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
            self._get_label(0),
        )

        self.label.set_fontsize(14)
        self.label.set_fontweight("normal")
        self.label.set_color("#666666")

        self.fig.subplots_adjust(left=0, right=1, bottom=0.1, top=1)

    @staticmethod
    def _get_label(timestep: int) -> str:
        """
        Generates a label for the environment.

        Parameters:
        timestep (int): The current time step.

        Returns:
        str: The generated label containing information about the environment and the current time step.
        """
        line1 = "Environment\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"

        return line1 + line2

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

        self.label.set_text(self._get_label(i))

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
