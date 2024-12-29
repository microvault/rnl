import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from mpl_toolkits.mplot3d import Axes3D, art3d
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler

from rnl.algorithms.rainbow import RainbowDQN
from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.engine.randomizer import TargetPosition
from rnl.engine.utils import (
    angle_to_goal,
    distance_to_goal,
    get_reward,
    min_laser,
    uniform_random,
    uniform_random_int,
)
from rnl.environment.generate_world import Generator
from rnl.environment.robot import Robot
from rnl.environment.sensor import SensorRobot
from shapely.geometry import Point


class NaviEnv(gym.Env):
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
        pretrained_model: bool,
    ):
        super().__init__()
        self.max_num_rays = 180
        state_size = (
            self.max_num_rays + 9
        )  # action_one_hot(7) + dist(1) + alpha(1) -> (action, distance, angle)
        self.action_space = spaces.Discrete(6)  # action
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        self.param = OmegaConf.load("./rnl/configs/limits.yaml")
        self.robot = Robot(robot_config)
        self.friction = env_config.friction
        self.space = self.robot.create_space()
        self.body = self.robot.create_robot(space=self.space, friction=self.friction)
        self.target_position = TargetPosition(
            window_size=1000,
            min_fraction=0.2,
            max_fraction=0.8,
            threshold=0.01,
            adjustment=0.05,
            episodes_interval=100
        )

        # -- Normalization -- #
        self.scaler_lidar = MinMaxScaler(feature_range=(0, 1))
        self.scaler_dist = MinMaxScaler(feature_range=(0, 1))
        self.scaler_alpha = MinMaxScaler(feature_range=(0, 1))
        self.scaler_reward = MinMaxScaler(feature_range=(0, 1))

        max_lidar, min_lidar = sensor_config.max_range, sensor_config.min_range
        self.scaler_lidar.fit(
            np.array(
                [
                    [min_lidar] * self.max_num_rays,
                    [max_lidar] * self.max_num_rays,
                ]
            )
        )

        self.reward_history = []

        self.xmax = env_config.grid_dimension - 0.25
        self.ymax = env_config.grid_dimension - 0.25
        self.dist_max = np.sqrt(self.xmax**2 + self.ymax**2)
        max_dist, min_dist = self.dist_max, 1.0
        self.scaler_dist.fit(np.array([[min_dist], [max_dist]]))

        max_alpha, min_alpha = 6.4, 0.0
        self.scaler_alpha.fit(np.array([[min_alpha], [max_alpha]]))

        max_reward, min_reward = 500.0, -500.0
        self.scaler_reward.fit(np.array([[min_reward], [max_reward]]))

        self.grid_lenght = env_config.grid_dimension

        if env_config.folder_map != "None":
            self.generator = Generator(
                random=env_config.porcentage_obstacles,
                mode=env_config.random_mode,
                folder=env_config.folder_map,
                name=env_config.name_map,
            )
            self.new_map_path, self.exterior, self.interior, self.segments, self.m, self.poly = self.generator.world(
                self.grid_lenght
            )
            self.sensor = SensorRobot(sensor_config, self.segments)
            self.initial_map = True
        else:
            self.generator = Generator(
                random=env_config.porcentage_obstacles,
                mode=env_config.random_mode,
                folder="None",
                name="None",
            )
            self.initial_map = False
            self.segments = []
            self.sensor = SensorRobot(sensor_config, map_segments=None)

        # -- Environmental parameters -- #
        self.max_lidar = sensor_config.max_range
        self.pretrained_model = pretrained_model
        self.random_state = env_config.random_mode
        self.data_collection = render_config.data_collection
        self.rgb_array = render_config.rgb_array
        self.max_timestep = env_config.timestep
        self.step_anim = env_config.timestep
        self.fps = render_config.fps
        self.threshold = robot_config.threshold
        self.controller = render_config.controller
        self.current_fraction = 0.5
        self.min_fraction = 0.2
        self.max_fraction = 0.8
        self.threshold = 0.01
        self.adjustment = 0.05

        # -- Local Variables -- #
        self.cumulated_reward = 0.0
        self.timestep = 0
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        self.last_position_x = 0
        self.last_position_y = 0
        self.last_theta = 0
        self.last_measurement = 0
        self.vl = 0.01
        self.vr = 0.01
        self.init_distance = 0
        self.action = 0
        self.scalar = 10# 1000
        self.randomization_frequency = env_config.randomization_interval
        self.epoch = 0
        self.current_rays = sensor_config.num_rays
        self.lidar_angle = np.linspace(0, 2 * np.pi, self.current_rays)
        self.measurement = np.zeros(self.current_rays)
        self.last_states = np.zeros(state_size)
        self.random_mode = env_config.random_mode

        if self.pretrained_model:
            self.rainbow = RainbowDQN.load(
                robot_config.path_model,
                device="cpu",
            )

        if self.rgb_array:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': '3d'})
            self.ax.remove()
            self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

            self.target = self.ax.plot3D(
                np.random.uniform(0, self.xmax),
                np.random.uniform(0, self.ymax),
                0,
                marker="x",
                markersize=6.0,
                color="red",
            )[0]

            self.agents = self.ax.plot3D(
                np.random.uniform(0, self.xmax),
                np.random.uniform(0, self.ymax),
                0,
                marker="o",
                markersize=6.0,
                color="orange",
            )[0]

            self.ani = animation.FuncAnimation

            self._init_animation(self.ax)
            if self.controller:
                print("Use the arrow keys to control the robot.")
                self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.reset()

    def on_key_press(self, event):
        if event.key == "up":
            self.action = 0
            self.vl = 0.05 * self.scalar
            self.vr = 0.0
        elif event.key == "down":
            self.action = 1
            self.vl = 0.1 * self.scalar
            self.vr = 0.0
        elif event.key == "left":
            self.action = 2
            self.vl = 0.05 * self.scalar
            self.vr = 0.15 * self.scalar
        elif event.key == "right":
            self.action = 3
            self.vl = 0.05 * self.scalar
            self.vr = -0.15 * self.scalar
        elif event.key == "w":
            self.action = 4
            self.vl = 0.01 * self.scalar
            self.vr = 0.0
        elif event.key == "d":
            self.action = 5
            self.vl = 0.05 * self.scalar
            self.vr = 0.3 * self.scalar
        elif event.key == "a":
            self.action = 6
            self.vl = 0.05 * self.scalar
            self.vr = -0.3 * self.scalar
        elif event.key == " ":
            self.vl = 0.0
            self.vr = 0.0

    def step_animation(self, i):
        if self.pretrained_model:
            self.action = self.rainbow.get_action(
                self.last_states, action_mask=None, training=True
            )[0]

        if not self.controller:
            if self.action == 0:
                self.vl = 0.05 * self.scalar
                self.vr = 0.0
            elif self.action == 1:
                self.vl = 0.1 * self.scalar
                self.vr = 0.0
            elif self.action == 2:
                self.vl = 0.05 * self.scalar
                self.vr = 0.15 * self.scalar
            elif self.action == 3:
                self.vl = 0.05 * self.scalar
                self.vr = -0.15 * self.scalar
            elif self.action == 4:
                self.vl = 0.01 * self.scalar
                self.vr = 0.0
            elif self.action == 5:
                self.vl = 0.05 * self.scalar
                self.vr = 0.3 * self.scalar
            elif self.action == 6:
                self.vl = 0.05 * self.scalar
                self.vr = -0.3 * self.scalar

        self.robot.move_robot(self.space, self.body, self.vl, self.vr)
        x, y, theta = (
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
        )

        intersections, lidar_measurements = self.sensor.sensor(
            x=x, y=y, theta=theta, segments=self.segments, max_range=self.max_lidar
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y)

        alpha = angle_to_goal(
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
            self.target_x,
            self.target_y,
        )

        diff_to_init = self.init_distance - dist

        collision, laser = min_laser(lidar_measurements, self.threshold)
        reward, done = get_reward(
            lidar_measurements, dist, collision, alpha, diff_to_init
        )

        self.current_rays = len(lidar_measurements)
        padded_lidar = np.zeros((self.max_num_rays,), dtype=np.float32)
        padded_lidar[: self.current_rays] = lidar_measurements[: self.current_rays]

        lidar_norm = self.scaler_lidar.transform(padded_lidar.reshape(1, -1)).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()

        action_one_hot = np.eye(7)[self.action]

        states = np.concatenate(
            (
                np.array(lidar_norm, dtype=np.float32),
                np.array(action_one_hot, dtype=np.int16),
                np.array(dist_norm, dtype=np.float32),
                np.array(alpha_norm, dtype=np.float32),
            )
        )

        self.cumulated_reward += reward
        self.reward_history.append(reward)
        self.last_states = states
        self._plot_anim(
            i,
            intersections,
            x,
            y,
            self.target_x,
            self.target_y,
            self.cumulated_reward,
            self.epoch,
        )

        self.space.step(1 / 60)

        truncated = self.timestep >= self.max_timestep

        if not self.poly.contains(Point(x, y)):
            done = True

        if done or truncated:
            self._stop()

    def step(self, action):

        if action == 0:
            self.vl = 0.05 * self.scalar
            self.vr = 0.0
        elif action == 1:
            self.vl = 0.1 * self.scalar
            self.vr = 0.0
        elif action == 2:
            self.vl = 0.05 * self.scalar
            self.vr = 0.15 * self.scalar
        elif action == 3:
            self.vl = 0.05 * self.scalar
            self.vr = -0.15 * self.scalar
        elif action == 4:
            self.vl = 0.01 * self.scalar
            self.vr = 0.0
        elif action == 5:
            self.vl = 0.05 * self.scalar
            self.vr = 0.3 * self.scalar
        elif action == 6:
            self.vl = 0.05 * self.scalar
            self.vr = -0.3 * self.scalar

        self.robot.move_robot(self.space, self.body, self.vl, self.vr)
        x, y, theta = (
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
        )

        intersections, lidar_measurements = self.sensor.sensor(
            x=x, y=y, theta=theta, segments=self.segments, max_range=self.max_lidar
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y)

        alpha = angle_to_goal(
            self.body.position.x,
            self.body.position.y,
            self.body.position.angle,
            self.target_x,
            self.target_y,
        )

        diff_to_init = self.init_distance - dist

        collision, laser = min_laser(lidar_measurements, self.threshold)
        reward, done = get_reward(
            lidar_measurements, dist, collision, alpha, diff_to_init
        )

        current_rays = len(lidar_measurements)
        padded_lidar = np.zeros((self.max_num_rays,), dtype=np.float32)
        padded_lidar[:current_rays] = lidar_measurements[:current_rays]

        lidar_norm = self.scaler_lidar.transform(padded_lidar.reshape(1, -1)).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()

        action_one_hot = np.eye(7)[action]

        states = np.concatenate(
            (
                np.array(lidar_norm, dtype=np.float32),
                np.array(action_one_hot, dtype=np.int16),
                np.array(dist_norm, dtype=np.float32),
                np.array(alpha_norm, dtype=np.float32),
            )
        )

        self.last_states = states
        self.last_measurement = lidar_measurements

        self.cumulated_reward += reward
        self.timestep += 1

        truncated = self.timestep >= self.max_timestep

        if not self.poly.contains(Point(x, y)):
            done = True

        self.space.step(1 / 60)

        return states, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.epoch += 1
        self.current_fraction = self.target_position.adjust_initial_distance(
            reward_history=self.reward_history,
            current_fraction=self.current_fraction,
            epoch=self.epoch
        )

        initial_distance = self.current_fraction * self.dist_max

        if self.epoch % self.randomization_frequency == 0:
            self.randomization()

        self.timestep = 0

        self.cumulated_reward = 0.0

        if not self.initial_map:
            self.new_map_path, self.exterior, self.interior, self.segments, self.m, self.poly = self.generator.world(
                self.grid_lenght
            )

        minx, miny, maxx, maxy = self.poly.bounds

        if self.rgb_array:
            for patch in self.ax.patches:
                patch.remove()

            self.ax.add_patch(self.new_map_path)
            art3d.pathpatch_2d_to_3d(self.new_map_path, z=0, zdir="z")

        x, y = self.random_point_in_poly(self.poly, minx, miny, maxx, maxy)

        self.target_x, self.target_y = self.random_point_in_poly_target(
            poly=self.poly,
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            center_x=x,
            center_y=y,
            distance=initial_distance
        )
        theta = np.random.uniform(0, 2 * np.pi)

        self.robot.reset_robot(self.body, x, y, theta)
        intersections, measurement = self.sensor.sensor(
            x=self.body.position.x,
            y=self.body.position.y,
            theta=self.body.position.angle,
            segments=self.segments,
            max_range=self.max_lidar,
        )

        self.last_measurement = measurement

        if self.rgb_array:
            self._plot_anim(
                0,
                intersections,
                self.body.position.x,
                self.body.position.y,
                self.target_x,
                self.target_y,
                self.cumulated_reward,
                self.epoch,
            )

        dist = distance_to_goal(
            self.body.position.x,
            self.body.position.y,
            self.target_x,
            self.target_y,
        )

        self.init_distance = dist

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
        # random action 0 to 6
        action = np.random.randint(0, 7)

        action_one_hot = np.eye(7)[action]

        self.last_states = np.concatenate(
            (
                np.array(lidar_norm, dtype=np.float32),
                np.array(action_one_hot, dtype=np.int16),
                np.array(dist_norm, dtype=np.float32),
                np.array(alpha_norm, dtype=np.float32),
            )
        )

        info = {}
        return self.last_states, info

    def render(self, mode="human"):
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
        # ------ Create wordld ------ #

        if not self.initial_map:
            self.new_map_path, _, _, _, m, poly = self.generator.world(self.grid_lenght)
            ax.set_xlim(0, self.grid_lenght)
            ax.set_ylim(0, self.grid_lenght)

        else:
            minx, miny, maxx, maxy = self.poly.bounds
            center_x = (minx + maxx) / 2.0
            center_y = (miny + maxy) / 2.0

            width = maxx - minx
            height = maxy - miny

            # Ajustar os limites para que o centro seja o do polígono
            ax.set_xlim(center_x - width/2, center_x + width/2)
            ax.set_ylim(center_y - height/2, center_y + height/2)

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
            0,
            0,
            0.05,
            self._get_label(0, 0, 0),
        )

        self.label.set_fontsize(14)
        self.label.set_fontweight("normal")
        self.label.set_color("#666666")

        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.20, top=0.95)

    @staticmethod
    def _get_label(timestep: int, score: float, episode: int) -> str:
        """
        Generates a label for the environment.

        Parameters:
        timestep (int): The current time step.

        Returns:
        str: The generated label containing information about the environment and the current time step.
        """
        line1 = "Environment\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        line3 = "Score:".ljust(14) + f"{score:4.0f}\n"
        line4 = "Episode:".ljust(14) + f"{episode:4.0f}\n"

        return line1 + line2 + line3 + line4

    def _plot_anim(
        self,
        i: int,
        intersections: np.ndarray,
        x: float,
        y: float,
        target_x: float,
        target_y: float,
        score: float,
        episode: int,
    ) -> None:

        self.label.set_text(self._get_label(i, score, episode))

        if hasattr(self, "laser_scatters"):
            for scatter in self.laser_scatters:
                scatter.remove()
            del self.laser_scatters

        self.laser_scatters = []
        for angle, intersection in zip(self.lidar_angle, intersections):
            if intersection is not None:
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

    def randomization(self):
        if self.random_mode == "hard":
            self.grid_lenght = uniform_random_int(
                self.param.environment.min_grid_dimension,
                self.param.environment.max_grid_dimension,
            )
            new_fov = uniform_random(
                self.param.sensor.min_fov, self.param.sensor.max_fov
            )
            new_num_rays = uniform_random_int(
                self.param.sensor.min_num_rays, self.param.sensor.max_num_rays
            )
            self.sensor.random_sensor(new_fov, new_num_rays)

            print("\n#------ New Random ----#")
            print(f"Grid lenght : {self.grid_lenght}")
            print(f"FOV : {new_fov}")
            print(f"Num rays : {new_num_rays}")

    def random_point_in_poly(self, poly, minx, miny, maxx, maxy, max_tries=1000):
        for _ in range(max_tries):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            if poly.contains(Point(x, y)):
                return x, y
        return (minx, miny)

    def random_point_in_poly_target(self, poly, minx, miny, maxx, maxy, center_x, center_y, distance, max_tries=1000):
        for _ in range(max_tries):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            if poly.contains(Point(x, y)):
                if np.sqrt((x - center_x)**2 + (y - center_y)**2) <= distance:
                    return x, y
        return (center_x, center_y)
