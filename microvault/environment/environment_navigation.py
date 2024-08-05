import os

import gymnasium as gym
import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
)
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_off_policy import train_off_policy
from agilerl.utils.utils import create_population, make_vect_envs
from gymnasium import spaces
from gymnasium.envs.registration import register
from mpl_toolkits.mplot3d import Axes3D, art3d
from tqdm import trange

from microvault.engine.collision import Collision
from microvault.engine.utils import (
    angle_to_goal,
    distance_to_goal,
    get_reward,
    min_laser,
)
from microvault.environment.generate_world import Generator
from microvault.environment.robot import Robot


class NaviEnv(gym.Env):
    def __init__(
        self,
        max_timestep: int = 2000,  # max step
        threshold: float = 0.05,  # 0.1 threshold
        grid_lenght: int = 5,  # TODO: error < 5 -> [5 - 15]
        rgb_array: bool = False,
        fps: int = 100,  # 10 frames per second
        state_size: int = 23,
        controller: bool = False,
    ):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )

        self.generator = Generator()
        self.collision = Collision()
        self.robot = Robot(self.collision)

        INIT_HP = {
            "BATCH_SIZE": 64,  # Batch size
            "LR": 0.0001,  # Learning rate
            "GAMMA": 0.99,  # Discount factor
            "MEMORY_SIZE": 1000000,  # Max memory buffer size
            "LEARN_STEP": 1,  # Learning frequency
            "N_STEP": 3,  # Step number to calculate td error
            "PER": True,  # Use prioritized experience replay buffer
            "ALPHA": 0.6,  # Prioritized replay buffer parameter
            "BETA": 0.4,  # Importance sampling coefficient
            "TAU": 0.001,  # For soft update of target parameters
            "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
            "NUM_ATOMS": 51,  # Unit number of support
            "V_MIN": -200.0,  # Minimum value of support
            "V_MAX": 200.0,  # Maximum value of support
            "NOISY": True,  # Add noise directly to the weights of the network
            "LEARNING_DELAY": 1000,  # Steps before starting learning
            "CHANNELS_LAST": False,  # Use with RGB states
            "TARGET_SCORE": 20.0,  # Target score that will beat the environment
            "MAX_STEPS": 1000000,  # Maximum number of steps an agent takes in an environment
            "EVO_STEPS": 10000,  # Evolution frequency
            "EVAL_STEPS": None,  # Number of evaluation steps per episode
            "EVAL_LOOP": 1,  # Number of evaluation episodes
        }
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print("device: ", device)

        net_config = {"arch": "mlp", "hidden_size": [400, 400]}

        # Define a Rainbow-DQN agent
        self.rainbow_dqn = RainbowDQN(
            state_dim=(self.observation_space.shape[0],),
            action_dim=self.action_space.n,
            one_hot=False,
            net_config=net_config,
            batch_size=INIT_HP["BATCH_SIZE"],
            lr=INIT_HP["LR"],
            learn_step=INIT_HP["LEARN_STEP"],
            gamma=INIT_HP["GAMMA"],
            tau=INIT_HP["TAU"],
            beta=INIT_HP["BETA"],
            n_step=INIT_HP["N_STEP"],
            device=device,
        )

        self.rainbow_dqn = RainbowDQN.load(
            "/Users/nicolasalan/microvault/microvault/checkpoints/model_final.pt",
            device=device,
        )

        self.states = np.zeros(state_size)

        self.rgb_array = rgb_array
        self.timestep = 0
        self.max_timestep = max_timestep
        self.step_anim = max_timestep
        self.fps = fps
        self.threshold = threshold
        self.grid_lenght = grid_lenght
        self.xmax = grid_lenght - 0.25
        self.ymax = grid_lenght - 0.25

        self.segments = []
        self.controller = controller
        self.cumulated_reward = 0.0

        self.target_x = 0
        self.target_y = 0
        self.last_position_x = 0
        self.last_position_y = 0
        self.last_theta = 0
        self.vl = 0.01
        self.vr = 0.01
        self.init_distance = 0

        self.lidar_angle = np.linspace(0, 2 * np.pi, 20)
        self.measurement = np.zeros(20)

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
            if controller:
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
        elif event.key == " ":
            self.vl = 0.0
            self.vr = 0.0

    def step_animation(self, i):
        if self.controller:
            predict = 1
        else:
            predict, *_ = self.rainbow_dqn.get_action(self.states, training=False)

            if predict == 0:
                self.vl = 0.05
                self.vr = 0.0
            elif predict == 1:
                self.vl = 0.1
                self.vr = 0.0
            elif predict == 2:
                self.vl = 0.05
                self.vr = 0.15
            elif predict == 3:
                self.vl = 0.05
                self.vr = -0.15

        x, y, theta = self.robot.move_robot(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            self.vl,
            self.vr,
        )

        intersections, measurement = self.robot.sensor(x=x, y=y, segments=self.segments)

        self._plot_anim(
            i,
            intersections,
            x,
            y,
            self.target_x,
            self.target_y,
        )

        dist = distance_to_goal(x, y, self.target_x, self.target_y)

        alpha = angle_to_goal(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            self.target_x,
            self.target_y,
        )

        self.states = np.concatenate(
            (
                np.array(measurement, dtype=np.float32),
                np.array([predict], dtype=np.float32),
                np.array([dist], dtype=np.float32),
                np.array([alpha], dtype=np.float32),
            )
        )

        self.last_theta = theta
        self.last_position_x = x
        self.last_position_y = y

        diff_to_init = self.init_distance - dist

        collision, laser = min_laser(measurement, self.threshold)
        reward, done = get_reward(measurement, dist, collision)

        self.cumulated_reward += reward

        self.timestep += 1

        truncated = self.timestep >= self.max_timestep

        print(
            "\rReward: {:.2f}\tC. reward: {:.2f}\tDistance: {:.2f}\tAngle: {:.2f}\tAction: {:.2f}".format(
                reward, self.cumulated_reward, dist, alpha, predict
            ),
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

        x, y, theta = self.robot.move_robot(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            self.vl,
            self.vr,
        )

        intersections, measurement = self.robot.sensor(x, y, self.segments)

        dist = distance_to_goal(
            self.last_position_x, self.last_position_y, self.target_x, self.target_y
        )

        alpha = angle_to_goal(
            self.last_position_x,
            self.last_position_y,
            self.last_theta,
            self.target_x,
            self.target_y,
        )

        self.states = np.concatenate(
            (
                np.array(measurement, dtype=np.float32),
                np.array([action], dtype=np.float32),
                np.array([dist], dtype=np.float32),
                np.array([alpha], dtype=np.float32),
            )
        )
        self.last_theta = theta
        self.last_position_x = x
        self.last_position_y = y

        diff_to_init = self.init_distance - dist

        collision, laser = min_laser(measurement, self.threshold)
        reward, done = get_reward(measurement, dist, collision)

        self.timestep += 1

        truncated = self.timestep >= self.max_timestep

        return self.states, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.timestep = 0

        self.vr = 0.02
        self.vl = 0.02

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

        self.states = np.concatenate(
            (
                np.array(measurement, dtype=np.float32),  # measurement
                np.array([0], dtype=np.float32),  # velocity
                np.array([dist], dtype=np.float32),  # distance
                np.array([alpha], dtype=np.float32),  # angle
            )
        )

        info = {}
        return self.states, info

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


register(
    id="NaviEnv-v0",
    entry_point="__main__:NaviEnv",  # Adjust this if NaviEnv is in a different module
    max_episode_steps=500000,
)


def main():
    INIT_HP = {
        "ALGO": "RainbowDQN",
        "BATCH_SIZE": 64,  # Batch size
        "LR": 0.0001,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 1000000,  # Max memory buffer size
        "LEARN_STEP": 1,  # Learning frequency
        "N_STEP": 3,  # Step number to calculate td error
        "PER": True,  # Use prioritized experience replay buffer
        "ALPHA": 0.6,  # Prioritized replay buffer parameter
        "BETA": 0.4,  # Importance sampling coefficient
        "TAU": 0.001,  # For soft update of target parameters
        "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
        "NUM_ATOMS": 51,  # Unit number of support
        "V_MIN": -200.0,  # Minimum value of support
        "V_MAX": 200.0,  # Maximum value of support
        "NOISY": True,  # Add noise directly to the weights of the network
        "LEARNING_DELAY": 1000,  # Steps before starting learning
        "CHANNELS_LAST": False,  # Use with RGB states
        "TARGET_SCORE": 100.0,  # Target score that will beat the environment
        "MAX_STEPS": 800000,  # Maximum number of steps an agent takes in an environment
        "EVO_STEPS": 10000,  # Evolution frequency
        "EVAL_STEPS": None,  # Number of evaluation steps per episode
        "EVAL_LOOP": 1,  # Number of evaluation episodes
        "POP_SIZE": 6,  # Population size
        "TOURN_SIZE": 2,  # Tournament size
        "ELITISM": True,  # Elitism in tournament selection
        "WANDB": False,  # Log with Weights and Biases
    }

    MUTATION_PARAMS = {
        # Relative probabilities
        "NO_MUT": 0.4,  # No mutation
        "ARCH_MUT": 0.2,  # Architecture mutation
        "NEW_LAYER": 0.2,  # New layer mutation
        "PARAMS_MUT": 0.2,  # Network parameters mutation
        "ACT_MUT": 0,  # Activation layer mutation
        "RL_HP_MUT": 0.2,  # Learning HP mutation
        "RL_HP_SELECTION": ["lr", "batch_size"],  # Learning HPs to choose from
        "MUT_SD": 0.1,  # Mutation strength
        "RAND_SEED": 1,  # Random seed
    }
    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [400, 400],  # Actor hidden size
    }

    num_envs = 16
    env = make_vect_envs("NaviEnv-v0", num_envs=num_envs)  # Create environment

    # Set-up
    device = "mps"
    one_hot = False
    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    # agent_pop = create_population(
    #     algo=INIT_HP['ALGO'],                 # Algorithm
    #     state_dim=state_dim,                  # State dimension
    #     action_dim=action_dim,                # Action dimension
    #     one_hot=one_hot,                      # One-hot encoding
    #     net_config=NET_CONFIG,                # Network configuration
    #     INIT_HP=INIT_HP,                      # Initial hyperparameters
    #     population_size=INIT_HP['POP_SIZE'],  # Population size
    #     num_envs=num_envs,                    # Number of vectorized environments
    #     device=device,
    # )

    # tournament = TournamentSelection(
    #     tournament_size=INIT_HP['TOURN_SIZE'], # Tournament selection size
    #     elitism=INIT_HP['ELITISM'],            # Elitism in tournament selection
    #     population_size=INIT_HP['POP_SIZE'],   # Population size
    #     eval_loop=INIT_HP['EVAL_LOOP'],        # Evaluate using last N fitness scores
    # )

    # mutations = Mutations(
    #     algo=INIT_HP['ALGO'],                                 # Algorithm
    #     no_mutation=MUTATION_PARAMS['NO_MUT'],                # No mutation
    #     architecture=MUTATION_PARAMS['ARCH_MUT'],             # Architecture mutation
    #     new_layer_prob=MUTATION_PARAMS['NEW_LAYER'],          # New layer mutation
    #     parameters=MUTATION_PARAMS['PARAMS_MUT'],             # Network parameters mutation
    #     activation=MUTATION_PARAMS['ACT_MUT'],                # Activation layer mutation
    #     rl_hp=MUTATION_PARAMS['RL_HP_MUT'],                   # Learning HP mutation
    #     rl_hp_selection=MUTATION_PARAMS['RL_HP_SELECTION'],   # Learning HPs to choose from
    #     mutation_sd=MUTATION_PARAMS['MUT_SD'],                # Mutation strength
    #     arch=NET_CONFIG['arch'],                              # Network architecture
    #     rand_seed=MUTATION_PARAMS['RAND_SEED'],               # Random seed
    #     device=device,
    # )

    # Define a Rainbow-DQN agent
    rainbow_dqn = RainbowDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        net_config=NET_CONFIG,
        batch_size=INIT_HP["BATCH_SIZE"],
        lr=INIT_HP["LR"],
        learn_step=INIT_HP["LEARN_STEP"],
        gamma=INIT_HP["GAMMA"],
        tau=INIT_HP["TAU"],
        beta=INIT_HP["BETA"],
        n_step=INIT_HP["N_STEP"],
        device=device,
    )
    field_names = ["state", "action", "reward", "next_state", "termination"]
    memory = PrioritizedReplayBuffer(
        memory_size=INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        num_envs=num_envs,
        alpha=INIT_HP["ALPHA"],
        gamma=INIT_HP["GAMMA"],
        device=device,
    )
    n_step_memory = MultiStepReplayBuffer(
        memory_size=INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        num_envs=num_envs,
        n_step=INIT_HP["N_STEP"],
        gamma=INIT_HP["GAMMA"],
        device=device,
    )

    trained_pop, pop_fitnesses = train_off_policy(
        env=env,
        env_name="NaviEnv-v0",
        algo=INIT_HP["ALGO"],
        pop=[rainbow_dqn],
        memory=memory,
        n_step_memory=n_step_memory,
        INIT_HP=INIT_HP,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        learning_delay=INIT_HP["LEARNING_DELAY"],
        target=INIT_HP["TARGET_SCORE"],
        n_step=True,
        per=True,
        tournament=None,
        mutation=None,
        wb=INIT_HP["WANDB"],  # Boolean flag to record run with Weights & Biases
        checkpoint=1000,
        checkpoint_path="model_dqn.pt",
    )

    save_path = "RainbowDQN.pt"
    rainbow_dqn.save_checkpoint(save_path)


if __name__ == "__main__":
    env = gym.make("NaviEnv-v0", rgb_array=True)

    env.reset()
    env.render()

    # main()
