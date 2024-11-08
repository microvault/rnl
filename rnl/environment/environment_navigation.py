import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from mpl_toolkits.mplot3d import Axes3D, art3d
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler

from rnl.algorithms.rainbow import RainbowDQN
from rnl.components.scale_dataset import ScaleDataset
from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.engine.collision import Collision
from rnl.engine.utils import (
    angle_to_goal,
    distance_to_goal,
    get_reward,
    min_laser,
    uniform_random_int,
)
from rnl.environment.generate_world import Generator
from rnl.environment.robot import Robot
from rnl.environment.sensor import SensorRobot


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

        state_size = sensor_config.num_rays + 9  # (action, distance, angle)
        self.action_space = spaces.Discrete(6)  # action
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        self.param = OmegaConf.load("./rnl/configs/limits.yaml")

        self.collision = Collision()
        self.robot = Robot(robot_config)
        self.sensor = SensorRobot(sensor_config)
        self.dataset = ScaleDataset()

        self.space = self.robot.create_space()
        self.body = self.robot.create_robot(self.space)

        # -- Normalization -- #
        self.scaler_lidar = MinMaxScaler(feature_range=(0, 1))
        self.scaler_dist = MinMaxScaler(feature_range=(0, 1))
        self.scaler_alpha = MinMaxScaler(feature_range=(0, 1))
        self.scaler_reward = MinMaxScaler(feature_range=(0, 1))

        max_lidar, min_lidar = sensor_config.max_range, sensor_config.min_range
        self.scaler_lidar.fit(
            np.array(
                [
                    [min_lidar] * sensor_config.num_rays,
                    [max_lidar] * sensor_config.num_rays,
                ]
            )
        )

        max_dist, min_dist = 12.0, 1.0
        self.scaler_dist.fit(np.array([[min_dist], [max_dist]]))

        max_alpha, min_alpha = 0.0, 6.4
        self.scaler_alpha.fit(np.array([[min_alpha], [max_alpha]]))

        max_reward, min_reward = 500.0, -500.0
        self.scaler_reward.fit(np.array([[min_reward], [max_reward]]))

        self.generator = Generator(
            env_config.porcentage_obstacles,
            env_config.random_mode,
        )

        # -- Environmental parameters -- #
        self.pretrained_model = pretrained_model
        self.random_state = env_config.random_mode
        self.data_collection = render_config.data_collection
        self.rgb_array = render_config.rgb_array
        self.max_timestep = env_config.timestep
        self.step_anim = env_config.timestep
        self.fps = render_config.fps
        self.threshold = robot_config.threshold
        self.grid_lenght = env_config.grid_dimension
        self.xmax = env_config.grid_dimension - 0.25
        self.ymax = env_config.grid_dimension - 0.25
        self.dist_max = np.sqrt(self.xmax**2 + self.ymax**2)
        self.controller = render_config.controller

        # -- Local Variables -- #
        self.segments = []
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
        self.scalar = 10
        self.randomization_frequency = env_config.randomization_interval
        self.epoch = 0
        self.lidar_angle = np.linspace(0, 2 * np.pi, 20)
        self.measurement = np.zeros(20)
        self.last_states = np.zeros(state_size)

        if self.pretrained_model:
            self.rainbow = RainbowDQN.load(
                robot_config.path_model,
                device="cpu",
            )

        if self.rgb_array:
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
                markersize=2.0,
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
            self.body.position.angle,
        )

        intersections, lidar_measurements = self.sensor.sensor(
            x=x, y=y, theta=theta, segments=self.segments
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y)

        alpha = angle_to_goal(
            self.robot.body.position.x,
            self.robot.body.position.y,
            self.robot.body.position.angle,
            self.target_x,
            self.target_y,
        )

        diff_to_init = self.init_distance - dist

        collision, laser = min_laser(lidar_measurements, self.threshold)
        reward, done = get_reward(
            lidar_measurements, dist, collision, alpha, diff_to_init
        )

        lidar_norm = self.scaler_lidar.transform(
            np.array(lidar_measurements).reshape(1, -1)
        ).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()

        # reward_norm = self.scaler_reward.transform(
        #     np.array(reward).reshape(1, -1)
        # ).flatten()

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

        if self.data_collection:
            self.dataset.store_step(
                state=self.last_states,
                action=self.action,
                reward=reward,
                next_state=states,
            )

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

        truncated = self.timestep >= self.max_timestep

        if done or truncated:
            self.dataset.save_dataset()
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
            self.body.position.angle,
        )

        intersections, lidar_measurements = self.sensor.sensor(
            x=x, y=y, theta=theta, segments=self.segments
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y)

        alpha = angle_to_goal(
            self.robot.body.position.x,
            self.robot.body.position.y,
            self.robot.body.position.angle,
            self.target_x,
            self.target_y,
        )

        diff_to_init = self.init_distance - dist

        collision, laser = min_laser(lidar_measurements, self.threshold)
        reward, done = get_reward(
            lidar_measurements, dist, collision, alpha, diff_to_init
        )

        lidar_norm = self.scaler_lidar.transform(
            np.array(lidar_measurements).reshape(1, -1)
        ).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()
        # reward_norm = self.scaler_reward.transform(
        #     np.array(reward).reshape(1, -1)
        # ).flatten()

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

        return states, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.epoch += 1

        if self.epoch % self.randomization_frequency == 0:
            self.randomization()

        self.timestep = 0

        self.cumulated_reward = 0.0

        new_map_path, exterior, interior, all_seg = self.generator.world(
            self.grid_lenght
        )
        self.segments = all_seg

        if self.rgb_array:
            for patch in self.ax.patches:
                patch.remove()

            self.ax.add_patch(new_map_path)
            art3d.pathpatch_2d_to_3d(new_map_path, z=0, zdir="z")

        while True:
            self.target_x = np.random.uniform(0, self.xmax)
            self.target_y = np.random.uniform(0, self.ymax)
            x = np.random.uniform(0, self.xmax)
            y = np.random.uniform(0, self.ymax)
            theta = np.random.uniform(0, 2 * np.pi)

            if self.collision.check_collision(
                exterior, interior, x, y
            ) and self.collision.check_collision(
                exterior, interior, self.target_x, self.target_y
            ):
                break

        self.robot.reset_robot(self.body, x, y, theta)
        intersections, measurement = self.sensor.sensor(
            x=self.robot.body.position.x,
            y=self.robot.body.position.y,
            theta=self.body.position.angle,
            segments=all_seg,
        )

        self.last_measurement = measurement

        if self.rgb_array:
            self._plot_anim(
                0,
                intersections,
                self.robot.body.position.x,
                self.robot.body.position.y,
                self.target_x,
                self.target_y,
                self.cumulated_reward,
                self.epoch,
            )

        dist = distance_to_goal(
            self.robot.body.position.x,
            self.robot.body.position.y,
            self.target_x,
            self.target_y,
        )

        self.init_distance = dist

        alpha = angle_to_goal(
            self.robot.body.position.x,
            self.robot.body.position.y,
            self.robot.body.position.angle,
            self.target_x,
            self.target_y,
        )

        lidar_norm = self.scaler_lidar.transform(
            np.array(measurement).reshape(1, -1)
        ).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()
        # reward_norm = self.scaler_reward.transform(
        #     np.array(0.0).reshape(1, -1)
        # ).flatten()

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

        path, _, _, _ = self.generator.world(self.grid_lenght)

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
            self._get_label(0, 0, 0),
        )

        self.label.set_fontsize(14)
        self.label.set_fontweight("normal")
        self.label.set_color("#666666")

        self.fig.subplots_adjust(left=0, right=1, bottom=0.1, top=1)

    def get_reward_function(self):

        return self.reward_function

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

    def randomization(self):
        self.grid_lenght = uniform_random_int(
            self.param.environment.min_grid_dimension,
            self.param.environment.max_grid_dimension,
        )
