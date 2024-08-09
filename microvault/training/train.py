import os

import gymnasium as gym
import imageio
import numpy as np
import torch

from microvault.environment.environment_navigation import NaviEnv
from microvault.hpo.mutation import Mutations
from microvault.hpo.tournament import TournamentSelection
from microvault.training.train_on_policy import train_on_policy
from microvault.training.utils import make_vect_envs, create_population
from tqdm import trange


def main():

    INIT_HP = {
        "ALGO": "PPO",  # Algorithm
        "DISCRETE_ACTIONS": True,  # Discrete action space
        "BATCH_SIZE": 64,  # Batch size
        "LR": 1e-3,  # Learning rate
        "LEARN_STEP": 128,  # Learning frequency
        "GAMMA": 0.99,  # Discount factor
        "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
        "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
        "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
        "ENT_COEF": 0.01,  # Entropy coefficient
        "VF_COEF": 0.5,  # Value function coefficient
        "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
        "TARGET_KL": None,  # Target KL divergence threshold
        "UPDATE_EPOCHS": 4,  # Number of policy update epochs
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,  # Population size
        "TOURN_SIZE": 2,  # Tournament size
        "ELITISM": True,  # Elitism in tournament selection
        "WANDB": False,  # Log with Weights and Biases
        "TARGET_SCORE": 100.0,  # Target score that will beat the environment
        "MAX_STEPS": 800000,  # Maximum number of steps an agent takes in an environment
        "EVO_STEPS": 10000,  # Evolution frequency
        "EVAL_STEPS": None,  # Number of evaluation steps per episode
        "EVAL_LOOP": 1,  # Number of evaluation episodes
    }

    MUTATION_PARAMS = {
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

    agent_pop = create_population(
        algo=INIT_HP["ALGO"],  # Algorithm
        state_dim=state_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=one_hot,  # One-hot encoding
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,  # Number of vectorized environments
        device=device,
    )

    tournament = TournamentSelection(
        tournament_size=INIT_HP["TOURN_SIZE"],  # Tournament selection size
        elitism=INIT_HP["ELITISM"],  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=INIT_HP["EVAL_LOOP"],  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        algo=INIT_HP["ALGO"],  # Algorithm
        no_mutation=MUTATION_PARAMS["NO_MUT"],  # No mutation
        architecture=MUTATION_PARAMS["ARCH_MUT"],  # Architecture mutation
        new_layer_prob=MUTATION_PARAMS["NEW_LAYER"],  # New layer mutation
        parameters=MUTATION_PARAMS["PARAMS_MUT"],  # Network parameters mutation
        activation=MUTATION_PARAMS["ACT_MUT"],  # Activation layer mutation
        rl_hp=MUTATION_PARAMS["RL_HP_MUT"],  # Learning HP mutation
        rl_hp_selection=MUTATION_PARAMS[
            "RL_HP_SELECTION"
        ],  # Learning HPs to choose from
        mutation_sd=MUTATION_PARAMS["MUT_SD"],  # Mutation strength
        arch=NET_CONFIG["arch"],  # Network architecture
        rand_seed=MUTATION_PARAMS["RAND_SEED"],  # Random seed
        device=device,
    )

    trained_pop, pop_fitnesses = train_on_policy(
        env=env,  # Gym-style environment
        env_name="NaviEnv-v0",  # Environment name
        algo=INIT_HP["ALGO"],  # Algorithm
        pop=agent_pop,  # Population of agents
        swap_channels=INIT_HP["CHANNELS_LAST"],  # Swap image channel from last to first
        max_steps=INIT_HP["MAX_STEPS"],  # Max number of training steps
        evo_steps=INIT_HP["EVO_STEPS"],  # Evolution frequency
        eval_steps=INIT_HP["EVAL_STEPS"],  # Number of steps in evaluation episode
        eval_loop=INIT_HP["EVAL_LOOP"],  # Number of evaluation episodes
        target=INIT_HP["TARGET_SCORE"],  # Target score for early stopping
        tournament=None,  # Tournament selection object
        mutation=None,  # Mutations object
        wb=INIT_HP["WANDB"],  # Boolean flag to record run with Weights & Biases
        checkpoint=100,
        checkpoint_path="model_ppo.pt",
    )


if __name__ == "__main__":
    gym.make("NaviEnv-v0")
    main()
