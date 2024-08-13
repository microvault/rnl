import gymnasium as gym

from microvault.algorithms.rainbow import RainbowDQN
from microvault.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
)

# from microvault.environment.environment_navigation import NaviEnv
from microvault.hpo.mutation import Mutations
from microvault.hpo.tournament import TournamentSelection
from microvault.training.train_off_policy import train_off_policy
from microvault.training.utils import create_population, make_vect_envs

MODE = "train"  # "eval"


def main():

    INIT_HP = {
        "ALGO": "Rainbow DQN",  # Algorithm
        "BATCH_SIZE": 64,  # Batch size
        "LR": 0.0001,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 1000000,  # Max memory buffer size
        "LEARN_STEP": 10,  # Learning frequency
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
        "TARGET_SCORE": 200.0,  # Target score that will beat the environment
        "MAX_STEPS": 800000,  # Maximum number of steps an agent takes in an environment
        "EVO_STEPS": 10000,  # Evolution frequency
        "EVAL_STEPS": None,  # Number of evaluation steps per episode
        "EVAL_LOOP": 1,  # Number of evaluation episodes
        "POP_SIZE": 6,  # Population size
        "TOURN_SIZE": 2,  # Tournament size
        "ELITISM": True,  # Elitism in tournament selection
        "WANDB": False,  # Log with Weights and Biases
    }

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [800, 600],  # Network hidden size
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

    num_envs = 16
    env = make_vect_envs("NaviEnv-v0", num_envs=num_envs)  # Create environment

    # Set-up
    device = "mps"
    one_hot = False
    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    rainbow = RainbowDQN(
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

    agent_pop = create_population(
        algo="RainbowDQN",  # Algorithm
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
        algo="Rainbow DQN",  # Algorithm
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
        arch="mlp",  # Network architecture
        rand_seed=MUTATION_PARAMS["RAND_SEED"],  # Random seed
        device=device,
    )

    print("Agent population: ", agent_pop)
    print("Tournament: ", tournament)
    print("Mutations: ", mutations)

    print("#---- Param ----#")
    print("Algo: ", INIT_HP["ALGO"])
    print("Batch Size: ", INIT_HP["BATCH_SIZE"])
    print("Learning Rate: ", INIT_HP["LR"])
    print("Learning Step: ", INIT_HP["LEARN_STEP"])
    print("Gamma: ", INIT_HP["GAMMA"])
    print("Population Size: ", INIT_HP["POP_SIZE"])
    print("Tournament Size: ", INIT_HP["TOURN_SIZE"])
    print("Elitism: ", INIT_HP["ELITISM"])
    print("WANDB: ", INIT_HP["WANDB"])
    print("Target Score: ", INIT_HP["TARGET_SCORE"])
    print("Max Steps: ", INIT_HP["MAX_STEPS"])
    print("Evo Steps: ", INIT_HP["EVO_STEPS"])
    print("Eval Steps: ", INIT_HP["EVAL_STEPS"])
    print("Eval Loop: ", INIT_HP["EVAL_LOOP"])
    print("Device: ", device)
    print("One hot: ", one_hot)
    print("Num envs: ", num_envs)
    print("State Size: ", env.single_observation_space.shape)
    print("Action size: ", env.single_action_space.n)

    trained_pop, pop_fitnesses = train_off_policy(
        env=env,
        env_name="NaviEnv-v0",
        algo=INIT_HP["ALGO"],
        pop=[rainbow],
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
        wb=False,
        checkpoint=100,
        checkpoint_path="RainbowDQN.pt",
    )


if __name__ == "__main__":

    if MODE == "train":
        gym.make("NaviEnv-v0")
        main()
    elif MODE == "eval":
        env = gym.make("NaviEnv-v0", rgb_array=True, controller=False)

        env.reset()
        env.render()
