from agilerl.components.replay_buffer import ReplayBuffer
from rnl.training.utils import make_vect_envs
from agilerl.utils.utils import create_population
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_off_policy import train_off_policy
import os
import torch
import multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb_key = os.environ.get("WANDB_API_KEY")

def main():
    INIT_PARAM = {
        'ENV_NAME': 'rnl',                  # Gym environment name
        'ALGO': 'Rainbow DQN',              # Algorithm
        'DOUBLE': True,                     # Use double Q-learning
        'CHANNELS_LAST': False,             # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'BATCH_SIZE': 256,                  # Batch size
        'LR': 1e-3,                         # Learning rate
        'MAX_STEPS': 1_000_000_00,           # Max no. steps
        'TARGET_SCORE': 500,                # Early training stop at avg score of last 100 episodes
        'GAMMA': 0.99,                      # Discount factor
        'TAU': 1e-3,                        # For soft update of target parameters
        'BETA': 0.4,                        # PER beta
        'PRIOR_EPS': 1e-6,                  # PER epsilon
        'MEMORY_SIZE': 1_000_000_00,            # Max memory buffer size
        'LEARN_STEP': 10,                    # Learning frequency
        'TAU': 1e-3,                        # For soft update of target parameters
        'TOURN_SIZE': 4,                    # Tournament size
        'ELITISM': True,                    # Elitism in tournament selection
        'POP_SIZE': 40,                      # Population size
        'EVO_STEPS': 10_000,                # Evolution frequency
        'EVAL_STEPS': None,                 # Evaluation steps
        'EVAL_LOOP': 10,                     # Evaluation episodes
        'LEARNING_DELAY': 1000,             # Steps before starting learning
        'WANDB': True,                      # Log with Weights and Biases
        'CHECKPOINT': 10_000,               # Checkpoint frequency
        'CHECKPOINT_PATH': 'checkpoints',   # Checkpoint path
        'SAVE_ELITE': True,                 # Save elite agent
        'ELITE_PATH': 'elite',              # Elite agent path
        'ACCELERATOR': None,                # Accelerator
        'VERBOSE': True,
        'TORCH_COMPILER': True,
        'EPS_START': 1.0,
        'EPS_END': 0.1,
        'EPS_DECAY': 0.995,
        'N_STEP': 3,
        'PER': True,
        'N_STEP_MEMORY': 1,
        'NUM_ATOMS': 51,
        'V_MIN': -200,
        'V_MAX': 200,
    }

    NET_CONFIG = {
        "arch": "mlp",                      # Network architecture
        "hidden_size": [800, 600],          # Actor hidden size
        "mlp_activation": "ReLU",
        "mlp_output_activation": "ReLU",
        "min_hidden_layers": 2,
        "max_hidden_layers": 4,
        "min_mlp_nodes": 64,
        "max_mlp_nodes": 500,
    }

    MUTATION_PARAMS = {
        'NO_MUT': 0.4,                                          # No mutation
        'ARCH_MUT': 0.2,                                        # Architecture mutation
        'NEW_LAYER': 0.2,                                       # New layer mutation
        'PARAMS_MUT': 0.2,                                      # Network parameters mutation
        'ACT_MUT': 0,                                           # Activation layer mutation
        'RL_HP_MUT': 0.2,                                       # Learning HP mutation
        'RL_HP_SELECTION': ["lr", "batch_size", "learn_step"],  # Learning HPs to choose from
        'MUT_SD': 0.1,                                          # Mutation strength
        'MIN_LR': 0.0001,
        'MAX_LR': 0.01,
        'MIN_LEARN_STEP': 1,
        'MAX_LEARN_STEP': 120,
        'MIN_BATCH_SIZE': 8,
        'MAX_BATCH_SIZE': 1024,
        'AGENTS_IDS': None,
        'MUTATE_ELITE': True,
        'RAND_SEED': 1,                                         # Random seed
        'ACTIVATION': ["ReLU", "ELU", "GELU"],                  # Activation functions to choose from
    }


    num_envs = 50
    env = make_vect_envs(
        num_envs=num_envs
    )
    one_hot = False
    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    agent_pop = create_population(
        algo=INIT_PARAM['ALGO'],                    # Algorithm
        state_dim=state_dim,                        # State dimension
        action_dim=action_dim,                      # Action dimension
        one_hot=one_hot,                            # One-hot encoding
        net_config=NET_CONFIG,                      # Network configuration
        INIT_HP=INIT_PARAM,                         # Initial hyperparameters
        population_size=INIT_PARAM['POP_SIZE'],     # Population size
        num_envs=num_envs,                          # Number of vectorized environments
        device=device,
        accelerator=INIT_PARAM['ACCELERATOR'],
        torch_compiler=INIT_PARAM['TORCH_COMPILER']
    )

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=INIT_PARAM['MEMORY_SIZE'],      # Max replay buffer size
        field_names=field_names,                    # Field names to store in memory
        device=device,
    )

    tournament = TournamentSelection(
        tournament_size=INIT_PARAM['TOURN_SIZE'], # Tournament selection size
        elitism=INIT_PARAM['ELITISM'],            # Elitism in tournament selection
        population_size=INIT_PARAM['POP_SIZE'],   # Population size
        eval_loop=INIT_PARAM['EVAL_LOOP'],        # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        algo=INIT_PARAM['ALGO'],                              # Algorithm
        no_mutation=MUTATION_PARAMS['NO_MUT'],                # No mutation
        architecture=MUTATION_PARAMS['ARCH_MUT'],             # Architecture mutation
        new_layer_prob=MUTATION_PARAMS['NEW_LAYER'],          # New layer mutation
        parameters=MUTATION_PARAMS['PARAMS_MUT'],             # Network parameters mutation
        activation=MUTATION_PARAMS['ACT_MUT'],                # Activation layer mutation
        rl_hp=MUTATION_PARAMS['RL_HP_MUT'],                   # Learning HP mutation
        rl_hp_selection=MUTATION_PARAMS['RL_HP_SELECTION'],   # Learning HPs to choose from
        mutation_sd=MUTATION_PARAMS['MUT_SD'],                # Mutation strength
        activation_selection=MUTATION_PARAMS['ACTIVATION'],   # Activation functions to choose from
        min_lr=MUTATION_PARAMS['MIN_LR'],
        max_lr=MUTATION_PARAMS['MAX_LR'],
        min_learn_step=MUTATION_PARAMS['MIN_LEARN_STEP'],
        max_learn_step=MUTATION_PARAMS['MAX_LEARN_STEP'],
        min_batch_size=MUTATION_PARAMS['MIN_BATCH_SIZE'],
        max_batch_size=MUTATION_PARAMS['MAX_BATCH_SIZE'],
        agent_ids=MUTATION_PARAMS['AGENTS_IDS'],
        mutate_elite=MUTATION_PARAMS['MUTATE_ELITE'],
        arch=NET_CONFIG['arch'],                              # Network architecture
        rand_seed=MUTATION_PARAMS['RAND_SEED'],               # Random seed
        device=device,
        accelerator=INIT_PARAM['ACCELERATOR']
    )

    trained_pop, pop_fitnesses = train_off_policy(
        env=env,                                        # Gym-style environment
        env_name=INIT_PARAM['ENV_NAME'],                # Environment name
        algo=INIT_PARAM['ALGO'],
        INIT_HP=INIT_PARAM,
        MUT_P=MUTATION_PARAMS,
        pop=agent_pop,                                  # Population of agents
        memory=memory,                                  # Replay buffer
        swap_channels=INIT_PARAM['CHANNELS_LAST'],      # Swap image channel from last to first
        max_steps=INIT_PARAM["MAX_STEPS"],              # Max number of training steps
        evo_steps=INIT_PARAM['EVO_STEPS'],              # Evolution frequency
        eval_steps=INIT_PARAM["EVAL_STEPS"],            # Number of steps in evaluation episode
        eval_loop=INIT_PARAM["EVAL_LOOP"],              # Number of evaluation episodes
        learning_delay=INIT_PARAM['LEARNING_DELAY'],    # Steps before starting learning
        eps_start=INIT_PARAM['EPS_START'],
        eps_end=INIT_PARAM['EPS_END'],
        eps_decay=INIT_PARAM['EPS_DECAY'],
        target=INIT_PARAM['TARGET_SCORE'],              # Target score for early stopping
        # n_step=INIT_PARAM['N_STEP'],                    # Use n-step returns
        # per=INIT_PARAM['PER'],                          # Use Prioritized Experience Replay
        # n_step_memory=INIT_PARAM['N_STEP_MEMORY'],      # n-step memory
        tournament=tournament,                          # Tournament selection object
        mutation=mutations,                             # Mutations object
        wb=INIT_PARAM['WANDB'],                         # Weights and Biases tracking
        checkpoint=INIT_PARAM['CHECKPOINT'],            # Checkpoint frequency
        checkpoint_path=INIT_PARAM['CHECKPOINT_PATH'],  # Checkpoint path
        save_elite=INIT_PARAM['SAVE_ELITE'],            # Save elite agent
        elite_path=INIT_PARAM['ELITE_PATH'],            # Elite agent path
        verbose=INIT_PARAM['VERBOSE'],                  # Verbose
        accelerator=INIT_PARAM['ACCELERATOR'],          # Accelerator
        wandb_api_key=wandb_key
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
