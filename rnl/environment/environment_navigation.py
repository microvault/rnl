import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from mpl_toolkits.mplot3d import Axes3D, art3d

from rnl.algorithms.rainbow import RainbowDQN
from rnl.configs.config import RobotConfig, SensorConfig
from rnl.engine.collision import Collision
from rnl.engine.utils import (
    angle_to_goal,
    distance_to_goal,
    get_reward,
    min_laser,
    normalize,
)

from rnl.environment.generate_world import Generator
from rnl.environment.robot import Robot


class NaviEnv(gym.Env):
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        state_size: int = 24,
        action_size: int = 6,
        max_timestep: int = 1000,  # max step
        threshold: float = 0.05,  # 0.1 threshold
        grid_lenght: int = 5,  # TODO: error < 5 -> [5 - 15]
        rgb_array: bool = False,
        fps: int = 100,  # 10 frames per second
        mode: str = "normal",
    ):
        super().__init__()
        self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )

        self.generator = Generator(grid_lenght=grid_lenght, mode=mode)
        self.collision = Collision()
        self.robot = Robot(
            self.collision,
            fov=sensor_config.fov,
            num_rays=sensor_config.num_rays,
            max_range=sensor_config.max_range,
            min_range=sensor_config.min_range,
        )

        self.last_states = np.zeros(state_size)

        self.rgb_array = rgb_array
        self.timestep = 0
        self.max_timestep = max_timestep
        self.step_anim = max_timestep
        self.fps = fps
        self.threshold = threshold
        self.grid_lenght = grid_lenght
        self.xmax = grid_lenght - 0.25
        self.ymax = grid_lenght - 0.25
        self.dist_max = np.sqrt(self.xmax**2 + self.ymax**2)

        self.segments = []
        # TODO
        self.controller = False
        self.cumulated_reward = 0.0

        self.target_x = 0
        self.target_y = 0
        self.last_position_x = 0
        self.last_position_y = 0
        self.last_theta = 0
        self.last_measurement = 0
        self.vl = 0.01
        self.vr = 0.01
        self.init_distance = 0

        self.lidar_angle = np.linspace(0, 2 * np.pi, 20)
        self.measurement = np.zeros(20)

        self.rainbow = RainbowDQN.load(
            "/Users/nicolasalan/microvault/rnl/checkpoints/model_dqn_5_90000.pt",
            device="mps",
        )

        if rgb_array:
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
                markersize=1.0,
                color="orange",
            )[0]

            self.ani = animation.FuncAnimation

            self._init_animation(self.ax)
            if self.controller:
                self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.reset()

    def on_key_press(self, event):
        if event.key == "up":
            self.vl = 0.05
            self.vr = 0.0
        elif event.key == "down":
            self.vl = 0.1
            self.vr = 0.0
        elif event.key == "left":
            self.vl = 0.05
            self.vr = 0.15
        elif event.key == "right":
            self.vl = 0.05
            self.vr = -0.15
        elif event.key == "w":
            self.vl = 0.01
            self.vr = 0.0
        elif event.key == "d":
            self.vl = 0.05
            self.vr = 0.3
        elif event.key == "a":
            self.vl = 0.05
            self.vr = -0.3
        elif event.key == " ":
            self.vl = 0.0
            self.vr = 0.0

    def step_animation(self, i):
        if self.controller:
            action = 1
        else:
            action = 1  # self.rainbow.get_action(self.last_states, training=False)

            if action == 0:
                self.vl = 0.05
                self.vr = 0.0
            elif action == 1:
                self.vl = 0.1
                self.vr = 0.0
            elif action == 2:
                self.vl = 0.05
                self.vr = 0.15
            elif action == 3:
                self.vl = 0.05
                self.vr = -0.15
            elif action == 4:
                self.vl = 0.01
                self.vr = 0.0
            elif action == 5:
                self.vl = 0.05
                self.vr = 0.3
            elif action == 6:
                self.vl = 0.05
                self.vr = -0.3

        self.robot.apply_forces(self.vl, self.vr)

        x, y, theta = self.robot.move_robot(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            self.vl,
            self.vr,
        )

        intersections, lidar_measurements = self.robot.sensor(
            x=x, y=y, segments=self.segments
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y)

        alpha = angle_to_goal(
            x,
            y,
            theta,
            self.target_x,
            self.target_y,
        )

        diff_to_init = self.init_distance - dist

        collision, laser = min_laser(lidar_measurements, self.threshold)
        reward, done = get_reward(
            lidar_measurements, dist, collision, alpha, diff_to_init
        )

        states = np.concatenate(
            (
                np.array(lidar_measurements, dtype=np.float32),
                np.array([action], dtype=np.int16),
                np.array([dist], dtype=np.float32),
                np.array([alpha], dtype=np.float32),
                np.array([reward], dtype=np.float32),
            )
        )

        # TODO: normalize states
        lidar_measurements_normalized = [
            normalize(lidar, 0.2, 6) for lidar in lidar_measurements
        ]
        dist_normalized = normalize(dist, 0.2, self.dist_max)
        angle_normalized = normalize(alpha, -np.pi, np.pi)
        reward_normalized = normalize(reward, -500, 500)

        states_normalized = np.concatenate(
            (
                np.array(lidar_measurements_normalized, dtype=np.float32),
                np.array([action], dtype=np.int16),
                np.array([dist_normalized], dtype=np.float32),
                np.array([angle_normalized], dtype=np.float32),
                np.array([reward_normalized], dtype=np.float32),
            )
        )

        self.last_theta = theta
        self.last_position_x = x
        self.last_position_y = y
        self.last_states = states
        self.last_measurement = lidar_measurements

        self.cumulated_reward += reward
        self.timestep += 1

        truncated = self.timestep >= self.max_timestep

        print(
            "\rReward: {:.2f}\tC. reward: {:.2f}\tDistance: {:.2f}\tAngle: {:.2f}\tAction: {:.2f}\tMin lidar: {:.2f}".format(
                states[23],
                self.cumulated_reward,
                states[21],
                states[22],
                states[20],
                np.min(states[:20]),
            ),
        )

        print(
            "------------------------------------------------------------------------------------------"
        )

        print(
            "\rNorm Reward: {:.2f}\tNorm  C. reward: {:.2f}\tNorm  Distance: {:.2f}\tNorm Angle: {:.2f}\tNorm  Action: {:.2f}\tNorm  Min lidar: {:.2f}".format(
                states_normalized[23],
                self.cumulated_reward,
                states_normalized[21],
                states_normalized[22],
                states_normalized[20],
                np.min(states_normalized[:20]),
            ),
        )

        self._plot_anim(
            i,
            intersections,
            x,
            y,
            self.target_x,
            self.target_y,
        )

        if done or truncated:
            self._stop()

    def step(self, action):

        if action == 0:
            self.vl = 0.05
            self.vr = 0.0
        elif action == 1:
            self.vl = 0.1
            self.vr = 0.0
        elif action == 2:
            self.vl = 0.05
            self.vr = 0.15
        elif action == 3:
            self.vl = 0.05
            self.vr = -0.15
        elif action == 4:
            self.vl = 0.01
            self.vr = 0.0
        elif action == 5:
            self.vl = 0.05
            self.vr = 0.3
        elif action == 6:
            self.vl = 0.05
            self.vr = -0.3

        x, y, theta = self.robot.move_robot(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            self.vl,
            self.vr,
        )

        intersections, lidar_measurements = self.robot.sensor(
            x=x, y=y, segments=self.segments
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y)

        alpha = angle_to_goal(
            x,
            y,
            theta,
            self.target_x,
            self.target_y,
        )

        diff_to_init = self.init_distance - dist

        collision, laser = min_laser(lidar_measurements, self.threshold)
        reward, done = get_reward(
            lidar_measurements, dist, collision, alpha, diff_to_init
        )

        states = np.concatenate(
            (
                np.array(lidar_measurements, dtype=np.float32),
                np.array([action], dtype=np.int16),
                np.array([dist], dtype=np.float32),
                np.array([alpha], dtype=np.float32),
                np.array([reward], dtype=np.float32),
            )
        )

        # dist
        # reward
        # action
        # measurement
        # diff measurement
        # probability

        self.last_theta = theta
        self.last_position_x = x
        self.last_position_y = y
        self.last_states = states
        self.last_measurement = lidar_measurements

        self.cumulated_reward += reward
        self.timestep += 1

        truncated = self.timestep >= self.max_timestep

        return states, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.timestep = 0

        self.vr = 0.05
        self.vl = 0.0

        self.cumulated_reward = 0.0

        new_map_path, exterior, interior, all_seg = self.generator.world()
        self.segments = all_seg

        if self.rgb_array:
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

        self.last_measurement = measurement

        if self.rgb_array:
            self._plot_anim(
                0,
                intersections,
                self.last_position_x,
                self.last_position_y,
                self.target_x,
                self.target_y,
            )

        dist = distance_to_goal(
            self.last_position_x, self.last_position_y, self.target_x, self.target_y
        )

        self.init_distance = dist

        alpha = angle_to_goal(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            self.target_x,
            self.target_y,
        )

        self.last_states = np.concatenate(
            (
                np.array(measurement, dtype=np.float32),  # measurement
                np.array([0], dtype=np.float32),  # velocity
                np.array([dist], dtype=np.float32),  # distance
                np.array([alpha], dtype=np.float32),  # angle
                np.array([0], dtype=np.float32),  # reward
            )
        )

        info = {}
        return self.last_states, info

    def render(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.step_animation,
            init_func=self.reset,
            blit=False,
            frames=self.step_anim,
            interval=self.fps,
        )

        plt.show()

    def _stop(self):
        self.reset()
        self.ani.frame_seq = self.ani.new_frame_seq()
        self.step_anim = self.max_timestep

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
