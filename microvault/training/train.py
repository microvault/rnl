import os

import gymnasium as gym
import imageio
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
from agilerl.utils.utils import make_vect_envs
from tqdm import trange

env = gym.make("NaviEnv-v0")

INIT_HP = {
    "BATCH_SIZE": 128,  # Batch size
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
    # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    "LEARNING_DELAY": 1000,  # Steps before starting learning
    "CHANNELS_LAST": False,  # Use with RGB states
    "TARGET_SCORE": 20.0,  # Target score that will beat the environment
    "MAX_STEPS": 1000000,  # Maximum number of steps an agent takes in an environment
    "EVO_STEPS": 10000,  # Evolution frequency
    "EVAL_STEPS": None,  # Number of evaluation steps per episode
    "EVAL_LOOP": 1,  # Number of evaluation episodes
}

num_envs = 16
env = make_vect_envs("NaviEnv-v0", num_envs=num_envs)  # Create environment
try:
    state_dim = env.single_observation_space.n  # Discrete observation space
    one_hot = True  # Requires one-hot encoding
except Exception:
    state_dim = env.single_observation_space.shape  # Continuous observation space
    one_hot = False  # Does not require one-hot encoding
try:
    action_dim = env.single_action_space.n  # Discrete action space
except Exception:
    action_dim = env.single_action_space.shape[0]  # Continuous action space

if INIT_HP["CHANNELS_LAST"]:
    # Adjust dimensions for PyTorch API (C, H, W), for envs with RGB image states
    state_dim = (state_dim[2], state_dim[0], state_dim[1])

# Set-up the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes
net_config = {"arch": "mlp", "hidden_size": [400, 400]}

# Define a Rainbow-DQN agent
rainbow_dqn = RainbowDQN(
    state_dim=state_dim,
    action_dim=action_dim,
    one_hot=one_hot,
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

mutations = Mutations(
    algo="RainbowDQN",  # Algorithm
    no_mutation=0.4,  # No mutation
    architecture=0.2,  # Architecture mutation
    new_layer_prob=0.2,  # New layer mutation
    parameters=0.2,  # Network parameters mutation
    activation=0,  # Activation layer mutation
    rl_hp=0.2,  # Learning HP mutation
    rl_hp_selection=["lr", "batch_size"],  # Learning HPs to choose from
    mutation_sd=0.1,  # Mutation strength
    arch=net_config["arch"],  # Network architecture
    rand_seed=1,  # Random seed
    device=device,
)

tournament = TournamentSelection(
    tournament_size=2,  # Tournament selection size
    elitism=True,  # Elitism in tournament selection
    population_size=6,  # Population size
    evo_step=1,
)  # Evaluate using last N fitness scores

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
    algo="RainbowDQN",
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
    tournament=tournament,
    mutation=mutations,
    wb=True,  # Boolean flag to record run with Weights & Biases
    checkpoint=INIT_HP["MAX_STEPS"],
    checkpoint_path="RainbowDQN.pt",
)

save_path = "RainbowDQN.pt"
rainbow_dqn.save_checkpoint(save_path)
