import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from agilerl.algorithms.dqn_rainbow import RainbowDQN
from gymnasium import spaces
from mpl_toolkits.mplot3d import Axes3D, art3d
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler

from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.engine.randomizer import (
    max_distance_in_polygon_with_holes,
    min_robot_goal_spawn_distance,
)
from rnl.engine.utils import (
    angle_to_goal,
    distance_to_goal,
    min_laser,
)
from rnl.environment.robot import Robot
from rnl.environment.sensor import SensorRobot
from rnl.environment.world import CreateWorld
from rnl.engine.rewards import get_reward
from rnl.engine.collisions import spawn_robot_and_goal


class NaviEnv(gym.Env):
    def __init__(
        self,
        robot_config: RobotConfig = RobotConfig(),
        sensor_config: SensorConfig = SensorConfig(),
        env_config: EnvConfig = EnvConfig(),
        render_config: RenderConfig = RenderConfig(),
        pretrained_model: bool = False,
        use_render: bool = False,
    ):
        super().__init__()
        self.max_num_rays = sensor_config.num_rays
        state_size = self.max_num_rays + 5
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        self.robot = Robot(robot_config)
        self.space = self.robot.create_space()
        self.body = self.robot.create_robot(space=self.space)

        self.generator = CreateWorld(
            folder=env_config.folder_map,
            name=env_config.name_map,
        )
        self.new_map_path, self.segments, self.poly = self.generator.world()
        self.sensor = SensorRobot(sensor_config, self.segments)

        # ------------ Normalization ------------ #
        self.scaler_lidar = MinMaxScaler(feature_range=(0, 1))
        self.scaler_dist = MinMaxScaler(feature_range=(0, 1))
        self.scaler_alpha = MinMaxScaler(feature_range=(0, 1))
        self.scaler_reward = MinMaxScaler(feature_range=(-10, 10))

        max_lidar, min_lidar = sensor_config.max_range, sensor_config.min_range
        self.scaler_lidar.fit(
            np.array(
                [
                    [min_lidar] * self.max_num_rays,
                    [max_lidar] * self.max_num_rays,
                ]
            )
        )
        self.use_render = use_render

        max_dist = max_distance_in_polygon_with_holes(self.poly)
        min_dist = min_robot_goal_spawn_distance(self.poly, 3.0, 3.0)
        self.scaler_dist.fit(np.array([[min_dist], [max_dist]]))

        max_alpha, min_alpha = 3.15, 0.0
        self.scaler_alpha.fit(np.array([[min_alpha], [max_alpha]]))

        max_reward, min_reward = 100.0, -100.0
        self.scaler_reward.fit(np.array([[max_reward], [min_reward]]))

        # -- Environmental parameters -- #
        self.max_lidar = sensor_config.max_range
        self.pretrained_model = pretrained_model
        self.max_timestep = env_config.timestep
        self.threshold = robot_config.threshold
        self.collision = robot_config.collision
        self.controller = render_config.controller

        # -- Local Variables -- #
        self.cumulated_reward: float = 0.0
        self.timestep: int = 0
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        self.last_position_x: float = 0.0
        self.last_position_y: float = 0.0
        self.last_theta: float = 0.0
        self.vl: float = 0.01
        self.vr: float = 0.01
        self.init_distance: float = 0.0
        self.action: int = 0
        self.scalar: int = 100
        self.current_fraction: float = 0.0
        self.current_rays = sensor_config.num_rays
        self.lidar_angle = np.linspace(0, 2 * np.pi, self.current_rays)
        self.measurement = np.zeros(self.current_rays)
        self.last_states = np.zeros(state_size)

        if self.pretrained_model:
            self.rainbow = RainbowDQN.load(
                robot_config.path_model,
                device="cpu",
            )

        if self.use_render:
            self.fig, self.ax = plt.subplots(
                1, 1, figsize=(6, 6), subplot_kw={"projection": "3d"}
            )
            self.ax.remove()
            self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

            self.target = self.ax.plot3D(
                np.random.uniform(0, 15),
                np.random.uniform(0, 15),
                0,
                marker="x",
                markersize=6.0,
                color="red",
            )[0]

            self.agents = self.ax.plot3D(
                np.random.uniform(0, 15),
                np.random.uniform(0, 15),
                0,
                marker="o",
                markersize=8.0,
                color="orange",
            )[0]

            self.ani = animation.FuncAnimation

            self._init_animation(self.ax)
            if self.controller:
                self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.reset()

    def on_key_press(self, event):
        if event.key == "up":
            self.action = 0
            self.vl = 0.05 * self.scalar
            self.vr = 0.0
        elif event.key == "right":
            self.action = 1
            self.vl = 0.05 * self.scalar
            self.vr = -0.015 * self.scalar
        elif event.key == "left":
            self.action = 2
            self.vl = 0.05 * self.scalar
            self.vr = 0.015 * self.scalar
        # Control and test
        elif event.key == " ":
            self.vl = 0.0
            self.vr = 0.0
        elif event.key == "r":
            self.vl = 0.0
            self.vr = -0.005 * self.scalar
        elif event.key == "e":
            self.vl = 0.0
            self.vr = 0.005 * self.scalar

    def step_animation(self, i):
        if self.pretrained_model:
            self.action = self.rainbow.get_action(
                self.last_states, action_mask=None, training=False
            )[0]

            if self.action == 0:
                self.vl = 0.05 * self.scalar
                self.vr = 0.0
            elif self.action == 1:
                self.vl = 0.05 * self.scalar
                self.vr = -0.015 * self.scalar
            elif self.action == 2:
                self.vl = 0.05 * self.scalar
                self.vr = 0.015 * self.scalar

        self.robot.move_robot(self.space, self.body, self.vl, self.vr)

        x, y, theta = (
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
        )
        intersections, lidar_measurements = self.sensor.sensor(
            x=x, y=y, theta=theta, max_range=self.max_lidar
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y)

        alpha = angle_to_goal(
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
            self.target_x,
            self.target_y,
        )

        collision, laser = min_laser(lidar_measurements, self.collision)

        padded_lidar = np.zeros((self.max_num_rays,), dtype=np.float32)
        padded_lidar[: self.current_rays] = lidar_measurements[: self.current_rays]

        lidar_norm = self.scaler_lidar.transform(padded_lidar.reshape(1, -1)).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()

        action_one_hot = np.eye(3)[self.action]

        states = np.concatenate(
            (
                np.array(lidar_norm, dtype=np.float32),
                np.array(action_one_hot, dtype=np.int16),
                np.array(dist_norm, dtype=np.float32),
                np.array(alpha_norm, dtype=np.float32),
            )
        )

        raw_reward, done = get_reward(
            distance=dist,
            collision=collision,
            alpha=alpha,
            distance_init=self.init_distance,
            step=i,
            time_penalty=0.01,
            threshold=self.threshold,
            scale_collision=1.0,
            scale_target=1.0,
            scale_orientation=1.0,
            scale_deadend=1.0,
            scale_global=1.0,
            scale_time=1.0,
        )

        raw_reward = np.clip(raw_reward, -100, 100)
        reward_norm = self.scaler_reward.transform([[raw_reward]]).flatten()

        # reward_norm = self.scaler_reward.transform(np.array(reward).reshape(1, -1)).flatten()

        min_lidar_norm = np.min(lidar_norm)

        self.last_states = states

        self._plot_anim(
            i,
            intersections,
            x,
            y,
            self.target_x,
            self.target_y,
            reward_norm[0],
            alpha_norm[0],
            min_lidar_norm,
            dist_norm[0],
            self.action,
        )

        self.space.step(1 / 60)

        truncated = self.timestep >= self.max_timestep

        if not self.poly.contains(Point(x, y)):
            done = True

        if done or truncated:
            self._stop()

    def step(self, action):

        if self.action == 0:
            self.vl = 0.05 * self.scalar
            self.vr = 0.0
        elif self.action == 1:
            self.vl = 0.05 * self.scalar
            self.vr = -0.015 * self.scalar
        elif self.action == 2:
            self.vl = 0.05 * self.scalar
            self.vr = 0.015 * self.scalar

        self.robot.move_robot(self.space, self.body, self.vl, self.vr)

        x, y, theta = (
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
        )
        intersections, lidar_measurements = self.sensor.sensor(
            x=x, y=y, theta=theta, max_range=self.max_lidar
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y)

        alpha = angle_to_goal(
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
            self.target_x,
            self.target_y,
        )

        collision, laser = min_laser(lidar_measurements, self.collision)

        reward, done = get_reward(
            measurement=lidar_measurements,
            distance=dist,
            collision=collision,
            alpha=alpha,
            distance_init=self.init_distance,
            step=self.timestep,
            time_penalty=0.01,
            threshold=self.threshold,
            scale_collision=1.0,
            scale_target=1.0,
            scale_orientation=1.0,
            scale_deadend=1.0,
            scale_global=1.0,
            scale_time=10.0,
        )


        padded_lidar = np.zeros((self.max_num_rays,), dtype=np.float32)
        padded_lidar[: self.current_rays] = lidar_measurements[: self.current_rays]

        lidar_norm = self.scaler_lidar.transform(padded_lidar.reshape(1, -1)).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()

        action_one_hot = np.eye(3)[self.action]

        states = np.concatenate(
            (
                np.array(lidar_norm, dtype=np.float32),
                np.array(action_one_hot, dtype=np.int16),
                np.array(dist_norm, dtype=np.float32),
                np.array(alpha_norm, dtype=np.float32),
            )
        )

        reward_norm = self.scaler_reward.transform(np.array(reward).reshape(1, -1)).flatten()

        self.last_states = states

        self.timestep += 1

        truncated = self.timestep >= self.max_timestep

        if not self.poly.contains(Point(x, y)):
            done = True

        self.space.step(1 / 60)

        return states, reward_norm, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.timestep = 0

        self.new_map_path, self.segments, self.poly = self.generator.world()

        minx, miny, maxx, maxy = self.poly.bounds

        if self.use_render:
            for patch in self.ax.patches:
                patch.remove()

            self.ax.add_patch(self.new_map_path)
            art3d.pathpatch_2d_to_3d(self.new_map_path, z=0, zdir="z")

        robot_pos, goal_pos = spawn_robot_and_goal(
            poly=self.poly, robot_clearance=self.threshold, goal_clearance=self.collision, min_robot_goal_dist=2.0
        )

        self.target_x, self.target_y = goal_pos[0], goal_pos[1]
        x, y = robot_pos[0], robot_pos[1]

        theta = np.random.uniform(0, 2 * np.pi)

        self.robot.reset_robot(self.body, x, y, theta)
        intersections, measurement = self.sensor.sensor(
            x=self.body.position.x,
            y=self.body.position.y,
            theta=self.body.position.angle,
            max_range=self.max_lidar,
        )

        dist = distance_to_goal(
            self.body.position.x,
            self.body.position.y,
            self.target_x,
            self.target_y,
        )


        alpha = angle_to_goal(
            self.body.position.x,
            self.body.position.y,
            self.body.position.angle,
            self.target_x,
            self.target_y,
        )

        self.current_rays = len(measurement)
        padded_lidar = np.zeros((self.max_num_rays,), dtype=np.float32)
        padded_lidar[: self.current_rays] = measurement[: self.current_rays]

        lidar_norm = self.scaler_lidar.transform(padded_lidar.reshape(1, -1)).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()
        action = np.random.randint(0, 3)

        action_one_hot = np.eye(3)[action]

        min_lidar_norm = np.min(lidar_norm)

        self.last_states = np.concatenate(
            (
                np.array(lidar_norm, dtype=np.float32),
                np.array(action_one_hot, dtype=np.int16),
                np.array(dist_norm, dtype=np.float32),
                np.array(alpha_norm, dtype=np.float32),
            )
        )

        self.init_distance = dist

        if self.use_render:
            self._plot_anim(
                0,
                intersections,
                self.body.position.x,
                self.body.position.y,
                self.target_x,
                self.target_y,
                0.0,
                alpha_norm,
                min_lidar_norm,
                dist_norm,
                self.action,
            )

        info = {}
        return self.last_states, info

    def render(self, mode="human"):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.step_animation,
            init_func=self.reset,
            blit=False,
            frames=self.max_timestep,
            interval=1 / 60,
        )

        plt.show()

    def _stop(self):
        self.reset()
        self.ani.frame_seq = self.ani.new_frame_seq()

    def _init_animation(self, ax: Axes3D) -> None:
        """
        Initializes the 3D animation by setting up the environment and camera parameters.

        Parameters:
        ax (Axes3D): The 3D axes to be used for plotting.

        Returns:
        None
        """
        # ------ Create wordld ------ #

        minx, miny, maxx, maxy = self.poly.bounds
        center_x = (minx + maxx) / 2.0
        center_y = (miny + maxy) / 2.0

        width = maxx - minx
        height = maxy - miny

        ax.set_xlim(center_x - width / 2, center_x + width / 2)
        ax.set_ylim(center_y - height / 2, center_y + height / 2)

        ax.add_patch(self.new_map_path)

        art3d.pathpatch_2d_to_3d(self.new_map_path, z=0, zdir="z")

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
        ax.elev = 40
        ax.azim = -255
        ax.dist = 200

        self.label = self.ax.text(
            0, 0, 0.001, self._get_label(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )

        self.label.set_fontsize(14)
        self.label.set_fontweight("normal")
        self.label.set_color("#666666")

        self.fig.subplots_adjust(left=0.05, right=0.55, bottom=0.05, top=0.95)

    @staticmethod
    def _get_label(
        timestep: int,
        score: float,
        state_angle,
        state_min_max_lidar,
        state_distance,
        action,
    ) -> str:
        """
        Generates a label for the environment.

        Parameters:
        timestep (int): The current time step.

        Returns:
        str: The generated label containing information about the environment and the current time step.
        """
        line1 = "Environment:\n"
        line2 = "Time Step:".ljust(14) + f"{timestep}\n"
        line3 = "Reward: ".ljust(14) + f"{score}\n"
        line4 = "Distance: ".ljust(14) + f"{state_distance}\n"
        line5 = "Angle:".ljust(14) + f"{state_angle}\n"
        line6 = "Lidar:".ljust(14) + f"{state_min_max_lidar}\n"
        line7 = "Action:".ljust(14) + f"{action}\n"

        return line1 + line2 + line3 + line4 + line5 + line6 + line7

    def _plot_anim(
        self,
        i: int,
        intersections: np.ndarray,
        x: float,
        y: float,
        target_x: float,
        target_y: float,
        score: float,
        state_angle: float,
        state_min_max_lidar: float,
        state_distance: float,
        action: int,
    ) -> None:

        self.label.set_text(
            self._get_label(
                i,
                score,
                state_angle,
                state_min_max_lidar,
                state_distance,
                action,
            )
        )

        if hasattr(self, "laser_scatters"):
            for scatter in self.laser_scatters:
                scatter.remove()
            del self.laser_scatters

        self.laser_scatters = []
        for angle, intersection in zip(self.lidar_angle, intersections):
            if intersection is not None and np.isfinite(intersection).all():
                if not (intersection[0] == 0 and intersection[1] == 0):
                    scatter = plt.scatter(
                        intersection[0], intersection[1], color="g", s=3.0
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

        if hasattr(self, "heading_line") and self.heading_line is not None:
            self.heading_line.remove()

        x2 = x + 2.0 * np.cos(self.body.angle)
        y2 = y + 2.0 * np.sin(self.body.angle)

        self.heading_line = self.ax.plot3D(
            [x, x2],
            [y, y2],
            [0, 0],
            color="red",
            linewidth=2,
        )[0]

        plt.draw()
