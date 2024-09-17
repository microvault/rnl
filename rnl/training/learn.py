from rnl.algorithms.rainbow import RainbowDQN
from rnl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
)
import gymnasium as gym
from rnl.configs.config import EnvConfig, RobotConfig, SensorConfig
from rnl.environment.environment_navigation import NaviEnv
from rnl.hpo.mutation import Mutations
from rnl.hpo.tournament import TournamentSelection
from rnl.training.train_off_policy import train_off_policy
from rnl.training.utils import create_population, make_vect_envs


def training(
    max_timestep,
    use_mutation,
    freq_evolution,
    log,
    batch_size,
    lr,
    pop_size,
    hidden_size,
    no_mut,
    arch_mut,
    new_layer,
    param_mut,
    act_mut,
    hp_mut,
    mut_strength,
    seed,
    num_envs,
    device,
    learn_step,
    n_step,
    memory_size,
    target_score,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    pretrained_model,
):

    env = make_vect_envs(
        env_name=NaviEnv,
        num_envs=num_envs,
        state_size=24,
        action_size=6,
        mode=env_config.mode,
        max_timestep=env_config.timestep,
        threshold=env_config.threshold,
        grid_lenght=env_config.grid_lenght,
        rgb_array=False,
        fps=env_config.fps,
        robot_config=robot_config,
        sensor_config=sensor_config,
    )

    # env = make_vect_envs("NaviEnv-v0", num_envs=num_envs)  # Create environment

    INIT_HP = {
        "ALGO": "Rainbow DQN",  # Algorithm
        "BATCH_SIZE": batch_size,  # Batch size
        "LR": lr,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": memory_size,  # Max memory buffer size
        "LEARN_STEP": learn_step,  # Learning frequency
        "N_STEP": n_step,  # Step number to calculate td error
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
        "TARGET_SCORE": target_score,  # Target score that will beat the environment
        "MAX_STEPS": max_timestep,  # Maximum number of steps an agent takes in an environment
        "EVO_STEPS": freq_evolution,  # Evolution frequency
        "EVAL_STEPS": None,  # Number of evaluation steps per episode
        "EVAL_LOOP": 1,  # Number of evaluation episodes
        "POP_SIZE": pop_size,  # Population size
        "TOURN_SIZE": 2,  # Tournament size
        "ELITISM": True,  # Elitism in tournament selection
        "WANDB": False,  # Log with Weights and Biases
    }

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": hidden_size,  # Network hidden size
    }

    MUTATION_PARAMS = {
        "NO_MUT": no_mut,  # No mutation
        "ARCH_MUT": arch_mut,  # Architecture mutation
        "NEW_LAYER": new_layer,  # New layer mutation
        "PARAMS_MUT": param_mut,  # Network parameters mutation
        "ACT_MUT": act_mut,  # Activation layer mutation
        "RL_HP_MUT": hp_mut,  # Learning HP mutation
        "RL_HP_SELECTION": ["lr", "batch_size"],  # Learning HPs to choose from
        "MUT_SD": mut_strength,  # Mutation strength
        "RAND_SEED": seed,  # Random seed
    }

    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    rainbow = RainbowDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=False,
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

    if use_mutation:

        agent_pop = create_population(
            algo="RainbowDQN",  # Algorithm
            state_dim=state_dim,  # State dimension
            action_dim=action_dim,  # Action dimension
            one_hot=False,  # One-hot encoding
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

    else:
        agent_pop = [rainbow]
        tournament = None
        mutations = None

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
    print("Num envs: ", num_envs)
    print("State Size: ", env.single_observation_space.shape)
    print("Action size: ", env.single_action_space.n)

    trained_pop, pop_fitnesses = train_off_policy(
        env=env,
        env_name="NaviEnv-v0",
        algo=INIT_HP["ALGO"],
        pop=agent_pop,
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
        wb=False,
        checkpoint=100,
        checkpoint_path="RainbowDQN.pt",
    )


def inference(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    pretrained_model=False,
):

    env = NaviEnv(robot_config, sensor_config, rgb_array=True, controller=True)

    env.reset()
    env.render()

    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n
