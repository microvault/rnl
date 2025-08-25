import os
from typing import Optional

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from gymnasium import spaces
from mpl_toolkits.mplot3d import Axes3D, art3d

from rnl.configs.config import EnvConfig, RenderConfig, RobotConfig, SensorConfig
from rnl.configs.rewards import RewardConfig
from rnl.engine.polygons import compute_polygon_diameter
from rnl.engine.spawn import spawn_robot_and_goal
from rnl.engine.utils import (
    CustomMinMaxScaler,
    angle_to_goal,
    distance_to_goal,
    load_pgm,
    min_laser,
)
from rnl.environment.generate import Generator
from rnl.environment.robot import Robot
from rnl.environment.sensor import SensorRobot
from rnl.network.policy import RNLPolicy


class NaviEnv(gym.Env):
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
        use_render: bool,
        type_reward: RewardConfig,
    ):
        super().__init__()
        self.max_num_rays = sensor_config.num_rays
        state_size = self.max_num_rays + 3
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        self.robot = Robot(robot_config)
        self.space = self.robot.create_space()
        self.body = self.robot.create_robot(space=self.space)

        self.max_lr = robot_config.max_vel_linear
        self.max_vr = robot_config.max_vel_angular

        self.reward_config = type_reward

        self.grid_length = 0
        self.poly = None
        self.infos_list = []
        self.steps_to_goal = 0
        self.steps_to_collision = 0
        self.steps_unsafe_area = 0
        self.steps_command_angular = 0
        self.pos_noise_std = 0.01
        self.ang_noise_std = 0.005
        self.lidar_noise_std = 0.02
        self.folder = env_config.folder_map
        self.name = env_config.name_map
        self.episode_id = 0

        self.generator = Generator(folder=self.folder, name=self.name)
        self.new_map_path, self.segments, self.poly = self.generator.world(
            grid_length=0, porcentage_obstacle=0
        )

        self.sensor = SensorRobot(sensor_config, self.segments)

        # ------------ Normalization ------------ #
        self.scaler_lidar = CustomMinMaxScaler(feature_range=(0, 1))
        self.scaler_dist = CustomMinMaxScaler(feature_range=(0, 1))
        self.scaler_alpha = CustomMinMaxScaler(feature_range=(0, 1))

        self.max_lidar, self.min_lidar = (
            sensor_config.max_range,
            sensor_config.min_range,
        )
        self.scaler_lidar.fit(
            np.array(
                [
                    [self.min_lidar] * self.max_num_rays,
                    [self.max_lidar] * self.max_num_rays,
                ]
            )
        )
        self.use_render = use_render
        self.max_dist = compute_polygon_diameter(self.poly) * 0.8
        self.min_dist = 0.0
        self.scaler_dist.fit(np.array([[self.min_dist], [self.max_dist]]))

        self.min_alpha, self.max_alpha = 0.0, 3.5 * 0.89
        self.scaler_alpha.fit(np.array([[self.min_alpha], [self.max_alpha]]))
        # -- Environmental parameters -- #
        self.max_lidar = sensor_config.max_range
        self.pretrained_model = robot_config.path_model
        self.max_timestep = env_config.timestep
        self.threshold = robot_config.threshold
        self.collision = robot_config.collision
        self.controller = render_config.controller

        # -- Local Variables -- #
        self.timestep: int = 0
        self.target_x: float = 0.0
        self.target_y: float = 0.0
        self.last_position_x: float = 0.0
        self.last_position_y: float = 0.0
        self.last_theta: float = 0.0
        self.vl: float = 0.01
        self.vr: float = 0.01
        self.action: int = 0
        self.initial_distance: float = 0.0
        self.scalar = env_config.scalar
        self.current_fraction: float = 0.0
        self.debug = render_config.debug
        self.plot = render_config.plot
        self.current_rays = sensor_config.num_rays
        self.lidar_angle = np.linspace(0, 2 * np.pi, self.current_rays)
        self.measurement = np.zeros(self.current_rays)
        self.last_states = np.zeros(state_size)

        self.policy = None
        if self.pretrained_model != "None":
            self.policy = RNLPolicy(
                in_dim=state_size, n_act=2, archive_path=robot_config.path_model
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

            if self.plot:
                self._init_reward_plot()

    def on_key_press(self, event):
        if event.key == "up":
            self.action = 0
            self.vl = self.max_lr / 2 * self.scalar
            self.vr = 0.0
        elif event.key == "right":
            self.action = 1
            self.vl = self.max_lr / 6 * self.scalar
            self.vr = +self.max_vr / 8 * self.scalar
        elif event.key == "left":
            self.action = 2
            self.vl = self.max_lr / 6 * self.scalar
            self.vr = -self.max_vr / 8 * self.scalar
        elif event.key == "down":
            self.action = 3
            self.vl = self.max_lr / 6 * self.scalar
            self.vr = 0.0 * self.scalar

        # Control and test
        elif event.key == " ":
            self.vl = 0.0
            self.vr = 0.0
        elif event.key == "r":
            self.vl = 0.0
            self.vr = (self.max_vr / 6) * self.scalar
        elif event.key == "e":
            self.vl = 0.0
            self.vr = -(self.max_vr / 6) * self.scalar

    def step_animation(self, i):
        if self.pretrained_model != "None" or not self.controller:
            if self.pretrained_model != "None" and self.policy is not None:
                self.policy.eval()
                action = self.policy.act(self.last_states)
                self.action = int(action)
            else:
                self.action = np.random.randint(0, 4)

        if not self.controller:
            if self.action == 0:
                self.vl = self.max_lr / 2 * self.scalar
                self.vr = 0.0
            elif self.action == 1:
                self.vl = self.max_lr / 6 * self.scalar
                self.vr = -self.max_vr / 8 * self.scalar
            elif self.action == 2:
                self.vl = self.max_lr / 6 * self.scalar
                self.vr = self.max_vr / 8 * self.scalar

        self.robot.move_robot(self.space, self.body, self.vl, self.vr)

        x, y, theta = (
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
        )

        intersections, lidar_measurements = self.sensor.sensor(
            x=x, y=y, theta=theta, max_range=self.max_lidar
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y, self.max_dist)

        alpha = angle_to_goal(
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
            self.target_x,
            self.target_y,
            self.max_alpha,
        )
        collision, laser = min_laser(lidar_measurements, self.collision)

        padded_lidar = np.zeros((self.max_num_rays,), dtype=np.float32)
        padded_lidar[: self.current_rays] = lidar_measurements[: self.current_rays]

        lidar_norm = self.scaler_lidar.transform(padded_lidar.reshape(1, -1)).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()

        action_one_hot = [np.int16(self.action != 0)]
        states = np.concatenate(
            (
                np.array(lidar_norm, dtype=np.float32),
                np.array(action_one_hot, dtype=np.int16),
                np.array(dist_norm, dtype=np.float32),
                np.array(alpha_norm, dtype=np.float32),
            )
        )

        if laser < (self.collision * 1.2):
            self.steps_unsafe_area += 1

        (
            collision_score,
            orientation_score,
            progress_score,
            time_score,
            obstacle,
            action_score,
            done,
        ) = self.reward_config.get_reward(
            lidar_measurements,
            poly=self.poly,
            position_x=x,
            position_y=y,
            current_distance=dist_norm[0],
            collision=collision,
            alpha=alpha_norm[0],
            threshold=self.threshold,
            threshold_collision=self.collision,
            min_distance=self.min_dist,
            max_distance=self.max_dist,
            action=self.action,
        )
        reward = (
            collision_score
            + orientation_score
            + progress_score
            + time_score
            + obstacle
            + action_score
        )

        min_lidar_norm = np.min(lidar_norm)

        self.last_states = states

        self._plot_anim(
            i,
            intersections,
            x,
            y,
            self.target_x,
            self.target_y,
            reward,
            orientation_score,
            progress_score,
            obstacle,
            alpha_norm[0],
            min_lidar_norm,
            dist_norm[0],
            self.action,
        )

        self.space.step(1 / 60)
        self.timestep += 1

        truncated = self.timestep >= self.max_timestep

        if collision and self.steps_to_collision == 0:
            self.steps_to_collision = self.timestep

        if reward == 1 and self.steps_to_goal == 0:
            self.steps_to_goal = self.timestep

        if self.action == 1 or self.action == 2:
            self.steps_command_angular += 1

        if self.plot:

            self.log_reward(
                obstacle,
                collision_score,
                orientation_score,
                progress_score,
                time_score,
                reward,
            )

        if done or truncated:
            if done or truncated:
                info = {
                    "steps_to_goal": self.steps_to_goal,
                    "steps_to_collision": self.steps_to_collision,
                    "steps_unsafe_area": self.steps_unsafe_area,
                    "steps_command_angular": self.steps_command_angular,
                    "total_timestep": self.timestep,
                }

                self.steps_to_goal = 0
                self.steps_to_collision = 0
                self.steps_unsafe_area = 0
                self.steps_command_angular = 0
                self.timestep = 0
            self.episode_starts = done
            self._stop()

    def step(self, action):
        vl = 0.0
        vr = 0.0

        if action == 0:
            vl = self.max_lr / 2 * self.scalar
            vr = 0.0
        elif action == 1:
            self.steps_command_angular += 1
            vl = self.max_lr / 6 * self.scalar
            vr = -self.max_vr / 8 * self.scalar
        elif action == 2:
            self.steps_command_angular += 1
            vl = self.max_lr / 6 * self.scalar
            vr = self.max_vr / 8 * self.scalar

        self.robot.move_robot(self.space, self.body, vl, vr)

        x, y, theta = (
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
        )

        intersections, lidar_measurements = self.sensor.sensor(
            x=x, y=y, theta=theta, max_range=self.max_lidar
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y, self.max_dist)

        alpha = angle_to_goal(
            self.body.position.x,
            self.body.position.y,
            self.body.angle,
            self.target_x,
            self.target_y,
            self.max_alpha,
        )

        collision_array, laser = min_laser(lidar_measurements, self.collision)
        collision = bool(np.any(collision_array))

        if laser < (self.collision * 1.2):
            self.steps_unsafe_area += 1

        padded_lidar = np.zeros((self.max_num_rays,), dtype=np.float32)
        padded_lidar[: self.current_rays] = lidar_measurements[: self.current_rays]

        lidar_norm = self.scaler_lidar.transform(padded_lidar.reshape(1, -1)).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()

        action_one_hot = [np.int16(action != 0)]

        states = np.concatenate(
            (
                np.array(lidar_norm, dtype=np.float16),
                np.array(action_one_hot, dtype=np.int16),
                np.array(dist_norm, dtype=np.float16),
                np.array(alpha_norm, dtype=np.float16),
            )
        )

        (
            collision_score,
            orientation_score,
            progress_score,
            time_score,
            obstacle,
            action_score,
            done,
        ) = self.reward_config.get_reward(
            lidar_measurements,
            poly=self.poly,
            position_x=x,
            position_y=y,
            current_distance=dist_norm[0],
            collision=collision,
            alpha=alpha_norm[0],
            threshold=self.threshold,
            threshold_collision=self.collision,
            min_distance=self.min_dist,
            max_distance=self.max_dist,
            action=action,
        )

        done = bool(done)

        reward = float(
            collision_score
            + orientation_score
            + progress_score
            + time_score
            + obstacle
            + action_score
        )
        self.last_states = states

        self.space.step(1 / 60)
        self.timestep += 1

        truncated = bool(self.timestep >= self.max_timestep)

        if self.debug:

            if collision and self.steps_to_collision == 0:
                self.steps_to_collision = self.timestep

            if reward == 1 and self.steps_to_goal == 0:
                self.steps_to_goal = self.timestep

            info = {
                "obstacle_score": obstacle,
                "orientation_score": orientation_score,
                "progress_score": progress_score,
                "time_score": time_score,
                "action": float(action),
                "dist": float(dist_norm[0]),
                "alpha": float(alpha_norm[0]),
                "min_lidar": float(min(lidar_norm)),
                "max_lidar": float(max(lidar_norm)),
            }
            if done or truncated:
                info.update(
                    {
                        "steps_to_goal": self.steps_to_goal,
                        "steps_to_collision": self.steps_to_collision,
                        "steps_unsafe_area": self.steps_unsafe_area,
                        "steps_command_angular": self.steps_command_angular,
                        "total_timestep": self.timestep,
                    }
                )

                self.steps_to_goal = 0
                self.steps_to_collision = 0
                self.steps_unsafe_area = 0
                self.steps_command_angular = 0
                self.timestep = 0

            self.infos_list.append(info)
            return states, reward, done, truncated, info

        else:
            return states, reward, done, truncated, {}

    def get_infos(self):
        infos = self.infos_list.copy()
        self.infos_list.clear()
        return infos

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        self.episode_id += 1

        try:
            self.new_map_path, self.segments, self.poly = self.generator.world(
                grid_length=0,
                porcentage_obstacle=0,
            )
            robot_pos, goal_pos = spawn_robot_and_goal(
                poly=self.poly,
                robot_clearance=self.threshold + 0.1,
                goal_clearance=self.collision + 0.1,
                min_robot_goal_dist=0.3,
            )

            self.target_x, self.target_y = goal_pos[0], goal_pos[1]
            x, y = robot_pos[0], robot_pos[1]

            self.sensor.update_map(self.segments)

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
                self.max_dist,
            )
            alpha = angle_to_goal(
                self.body.position.x,
                self.body.position.y,
                self.body.position.angle,
                self.target_x,
                self.target_y,
                self.max_alpha,
            )

            self.initial_distance = dist

            self.current_rays = len(measurement)
            padded_lidar = np.zeros((self.max_num_rays,), dtype=np.float32)
            padded_lidar[: self.current_rays] = measurement[: self.current_rays]

            lidar_norm = self.scaler_lidar.transform(
                padded_lidar.reshape(1, -1)
            ).flatten()
            dist_norm = self.scaler_dist.transform(
                np.array(dist).reshape(1, -1)
            ).flatten()
            alpha_norm = self.scaler_alpha.transform(
                np.array(alpha).reshape(1, -1)
            ).flatten()

            action = np.random.randint(0, 3)
            action_one_hot = [np.int16(action != 0)]
            min_lidar_norm = np.min(lidar_norm)

            states = np.concatenate(
                (
                    np.array(lidar_norm, dtype=np.float32),
                    np.array(action_one_hot, dtype=np.int16),
                    np.array(dist_norm, dtype=np.float32),
                    np.array(alpha_norm, dtype=np.float32),
                )
            )
            self.last_states = states

            if self.use_render:
                for patch in self.ax.patches:
                    patch.remove()
                self.ax.add_patch(self.new_map_path)
                art3d.pathpatch_2d_to_3d(self.new_map_path, z=0, zdir="z")

                self._plot_anim(
                    0,
                    intersections,
                    self.body.position.x,
                    self.body.position.y,
                    self.target_x,
                    self.target_y,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    alpha_norm,
                    min_lidar_norm,
                    dist_norm,
                    self.action,
                )

        except Exception as e:
            print(f"[RESET-ERROR] Error setting up the scenario: {e}")
            raise
        info = {}

        return states, info

    def render(self, mode="human", record: bool = False):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.step_animation,
            init_func=self.reset,
            blit=False,
            frames=self.max_timestep,
            interval=1 / 60,
        )
        plt.show()

    def _stop(self, test=None):
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
        thresh = 0.65
        yaml_path = self.folder + "/" + self.name + ".yaml"

        with open(yaml_path) as f:
            info = yaml.safe_load(f)

        res = float(info["resolution"])
        ox, oy, oyaw = info["origin"]
        pgm_path = os.path.join(os.path.dirname(yaml_path), info["image"])

        img = load_pgm(pgm_path)
        if info.get("negate", 0):
            img = 255 - img

        occ = img < int(thresh * 255)
        h, w = occ.shape
        gx, gy = w * res, h * res

        ax.set_xlim(ox, ox + gx)
        ax.set_ylim(oy, oy + gy)
        ax.invert_yaxis()

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

        if self.debug:
            self.label = self.ax.text(
                0.1, 0, 0.001, self._get_label(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
            )

            self.label.set_fontsize(14)
            self.label.set_fontweight("normal")
            self.label.set_color("#666666")

        self.fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95)

    @staticmethod
    def _get_label(
        timestep: int,
        score: float,
        score_angle: float,
        score_distance: float,
        score_obstacle: float,
        state_angle: float,
        state_min_max_lidar: float,
        state_distance: float,
        action: int,
    ) -> str:
        """
        Generates a label for the environment.

        Parameters:
        timestep (int): The current time step.

        Returns:
        str: The generated label containing information about the environment and the current time step.
        """
        if isinstance(state_distance, np.ndarray):
            state_distance = state_distance.item()
        if isinstance(state_angle, np.ndarray):
            state_angle = state_angle.item()
        if isinstance(state_min_max_lidar, np.ndarray):
            state_min_max_lidar = state_min_max_lidar.item()

        line1 = "Environment:\n"
        line2 = "Time Step:".ljust(14) + f"{timestep}\n"
        space1 = "------\n"
        line3 = "R. total.: ".ljust(14) + f"{score:.4f}\n"
        line4 = "R. angle: ".ljust(14) + f"{score_angle:.4f}\n"
        line5 = "R. dist.: ".ljust(14) + f"{score_distance:.4f}\n"
        line6 = "R. obst.: ".ljust(14) + f"{score_obstacle:.4f}\n"
        space2 = "------\n"
        line7 = "Distance: ".ljust(14) + f"{state_distance:.4f}\n"
        line8 = "Angle:".ljust(14) + f"{state_angle:.4f}\n"
        line9 = "Lidar:".ljust(14) + f"{state_min_max_lidar:.4f}\n"
        line10 = "Action:".ljust(14) + f"{action}\n"

        return (
            line1
            + line2
            + space1
            + line3
            + line4
            + line5
            + line6
            + space2
            + line7
            + line8
            + line9
            + line10
        )

    def _plot_anim(
        self,
        i: int,
        intersections: np.ndarray,
        x: float,
        y: float,
        target_x: float,
        target_y: float,
        score: float,
        score_angle: float,
        score_distance: float,
        score_obstacle: float,
        state_angle: float,
        state_min_max_lidar: float,
        state_distance: float,
        action: int,
    ) -> None:

        if self.debug:
            self.label.set_text(
                self._get_label(
                    i,
                    score,
                    score_angle,
                    score_distance,
                    score_obstacle,
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

        if intersections is not None and np.size(intersections) > 0:
            for angle, intersection in zip(self.lidar_angle, intersections):
                if intersection is not None and np.isfinite(intersection).all():
                    if not (intersection[0] == 0 and intersection[1] == 0):
                        scatter = plt.scatter(
                            intersection[0], intersection[1], color="g", s=3.0
                        )
                        self.laser_scatters.append(scatter)

        self.agents.set_data_3d([x], [y], [0])
        self.target.set_data_3d([target_x], [target_y], [0])

        if hasattr(self, "heading_line") and self.heading_line is not None:
            self.heading_line.remove()

        x2 = x + 0.1 * np.cos(self.body.angle)
        y2 = y + 0.1 * np.sin(self.body.angle)
        self.heading_line = self.ax.plot3D(
            [x, x2], [y, y2], [0, 0], color="red", linewidth=1
        )[0]

        plt.draw()

    def log_reward(
        self,
        obstacle_score: float,
        collision_score: float,
        orientation_score: float,
        progress_score: float,
        time_score: float,
        reward: float,
    ):
        self.obstacle_scores.append(obstacle_score)
        self.collision_scores.append(collision_score)
        self.orientation_scores.append(orientation_score)
        self.progress_scores.append(progress_score)
        self.time_scores.append(time_score)
        self.rewards.append(reward)

        current_step = len(self.rewards)

        self.obstacle_line.set_data(range(current_step), self.obstacle_scores)
        self.collision_line.set_data(range(current_step), self.collision_scores)
        self.orientation_line.set_data(range(current_step), self.orientation_scores)
        self.progress_line.set_data(range(current_step), self.progress_scores)
        self.time_line.set_data(range(current_step), self.time_scores)
        self.total_reward_line.set_data(range(current_step), self.rewards)

        if current_step > self.reward_ax.get_xlim()[1]:
            self.reward_ax.set_xlim(0, current_step + 10)

        all_rewards = (
            self.obstacle_scores
            + self.collision_scores
            + self.orientation_scores
            + self.progress_scores
            + self.time_scores
            + self.rewards
        )
        min_y = min(all_rewards) if all_rewards else -1
        max_y = max(all_rewards) if all_rewards else 1
        self.reward_ax.set_ylim(min_y - 1, max_y + 1)

        self.reward_ax.figure.canvas.draw()
        self.reward_ax.figure.canvas.flush_events()

    def _init_reward_plot(self):
        """Initializes the real-time reward plot."""
        self.reward_fig, self.reward_ax = plt.subplots(figsize=(8, 6))
        self.reward_ax.set_title("Real-Time Reward Components")
        self.reward_ax.set_xlabel("Steps")
        self.reward_ax.set_ylabel("Reward")

        # Initialize lists to store rewards
        self.obstacle_scores = []
        self.collision_scores = []
        self.orientation_scores = []
        self.progress_scores = []
        self.time_scores = []
        self.rewards = []

        # Initialize lines for each reward component
        (self.obstacle_line,) = self.reward_ax.plot([], [], label="Obstacle Score")
        (self.collision_line,) = self.reward_ax.plot([], [], label="Collision Score")
        (self.orientation_line,) = self.reward_ax.plot(
            [], [], label="Orientation Score"
        )
        (self.progress_line,) = self.reward_ax.plot([], [], label="Progress Score")
        (self.time_line,) = self.reward_ax.plot([], [], label="Time Score")
        (self.total_reward_line,) = self.reward_ax.plot(
            [], [], label="Total Reward", linewidth=2, color="black"
        )

        self.reward_ax.legend(loc="upper left")
        self.reward_ax.set_xlim(0, 100)
        self.reward_ax.set_ylim(-1, 10)

        self.reward_fig.tight_layout()
