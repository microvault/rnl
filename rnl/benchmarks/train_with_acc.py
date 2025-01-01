from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.replay_data import ReplayDataset
from agilerl.components.sampler import Sampler
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from rnl.training.utils import make_vect_envs
from accelerate import Accelerator
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import trange
import multiprocessing as mp

def main():
    accelerator = Accelerator()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("===== AgileRL Online Distributed Demo =====")
    accelerator.wait_for_everyone()

    INIT_PARAM = {
        'ENV_NAME': 'rnl',                  # Gym environment name
        'ALGO': 'Rainbow DQN',              # Algorithm
        'DOUBLE': True,                     # Use double Q-learning
        'CHANNELS_LAST': False,             # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'BATCH_SIZE': 128,                  # Batch size
        'LR': 1e-3,                         # Learning rate
        'MAX_STEPS': 1_000_000_0,             # Max no. steps
        'TARGET_SCORE': 500,               # Early training stop at avg score of last 100 episodes
        'GAMMA': 0.99,                      # Discount factor
        'TAU': 1e-3,                        # For soft update of target parameters
        'BETA': 0.4,                        # PER beta
        'PRIOR_EPS': 1e-6,                  # PER epsilon
        'MEMORY_SIZE': 1000000,               # Max memory buffer size
        'LEARN_STEP': 1,                    # Learning frequency
        'TAU': 1e-3,                        # For soft update of target parameters
        'TOURN_SIZE': 4,                    # Tournament size
        'ELITISM': True,                    # Elitism in tournament selection
        'POP_SIZE': 40,                      # Population size
        'EVO_STEPS': 10_000,                # Evolution frequency
        'EVAL_STEPS': None,                 # Evaluation steps
        'EVAL_LOOP': 1,                     # Evaluation episodes
        'LEARNING_DELAY': 1000,             # Steps before starting learning
        'WANDB': False,                      # Log with Weights and Biases
        'CHECKPOINT': 10_000,               # Checkpoint frequency
        'CHECKPOINT_PATH': 'checkpoints',   # Checkpoint path
        'SAVE_ELITE': True,                 # Save elite agent
        'ELITE_PATH': 'elite',              # Elite agent path
        'ACCELERATOR': accelerator,                # Accelerator
        'VERBOSE': True,
        'TORCH_COMPILER': False,
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

    num_envs = 25
    env = make_vect_envs(
        num_envs=num_envs
    )
    one_hot = False
    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    pop = create_population(
        algo=INIT_PARAM['ALGO'],                    # Algorithm
        state_dim=state_dim,                        # State dimension
        action_dim=action_dim,                      # Action dimension
        one_hot=one_hot,                            # One-hot encoding
        net_config=NET_CONFIG,                      # Network configuration
        INIT_HP=INIT_PARAM,                         # Initial hyperparameters
        population_size=INIT_PARAM['POP_SIZE'],     # Population size
        num_envs=num_envs,                          # Number of vectorized environments
        accelerator=INIT_PARAM['ACCELERATOR'],
        torch_compiler=INIT_PARAM['TORCH_COMPILER']
    )

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=INIT_PARAM['MEMORY_SIZE'],  # Max replay buffer size
        field_names=field_names,
    )
    replay_dataset = ReplayDataset(memory, INIT_PARAM["BATCH_SIZE"])
    replay_dataloader = DataLoader(replay_dataset, batch_size=None)
    replay_dataloader = accelerator.prepare(replay_dataloader)
    sampler = Sampler(
        distributed=True, dataset=replay_dataset, dataloader=replay_dataloader
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
        accelerator=INIT_PARAM['ACCELERATOR']
    )

    max_steps = INIT_PARAM['MAX_STEPS']
    learning_delay = INIT_PARAM['LEARNING_DELAY']

    # Exploration params
    eps_start = INIT_PARAM['EPS_START']
    eps_end = INIT_PARAM['EPS_END']
    eps_decay = INIT_PARAM['EPS_DECAY']
    epsilon = eps_start

    evo_steps = INIT_PARAM['EVO_STEPS']
    eval_steps = INIT_PARAM['EVAL_STEPS']
    eval_loop = INIT_PARAM['EVAL_LOOP']

    total_steps = 0

    accel_temp_models_path = "models/{}".format("LunarLander-v2")
    if accelerator.is_main_process:
        if not os.path.exists(accel_temp_models_path):
            os.makedirs(accel_temp_models_path)

    print(f"\nDistributed training on {accelerator.device}...")

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step", disable=not accelerator.is_local_main_process)
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        accelerator.wait_for_everyone()
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores, losses = [], []
            steps = 0
            epsilon = eps_start

            for idx_step in range(evo_steps):
                # Get next action from agent
                action = agent.get_action(state, epsilon)
                epsilon = max(
                    eps_end, epsilon * eps_decay
                )  # Decay epsilon for exploration

                # Act in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                scores += np.array(reward)
                steps += num_envs
                total_steps += num_envs

                # Collect scores for completed episodes
                for idx, (d, t) in enumerate(zip(terminated, truncated)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0

                # Save experience to replay buffer
                memory.save_to_memory_vect_envs(
                    state, action, reward, next_state, terminated
                )

                # Learn according to learning frequency
                if memory.counter > learning_delay and len(memory) >= agent.batch_size:
                    for _ in range(num_envs // agent.learn_step):
                        # Sample dataloader
                        experiences = sampler.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)

                state = next_state

            pbar.update(evo_steps // len(pop))
            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Reset epsilon start to latest decayed value for next round of population training
        eps_start = epsilon

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=INIT_PARAM["CHANNELS_LAST"],
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        if accelerator.is_main_process:
            print(f"--- Global steps {total_steps} ---")
            print(f"Steps {[agent.steps[-1] for agent in pop]}")
            print(f"Scores: {mean_scores}")
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(
                f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
            )

        # Tournament selection and population mutation
        accelerator.wait_for_everyone()
        for model in pop:
            model.unwrap_models()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)
            for pop_i, model in enumerate(pop):
                model.save_checkpoint(f"{accel_temp_models_path}/DQN_{pop_i}.pt")
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            for pop_i, model in enumerate(pop):
                model.load_checkpoint(f"{accel_temp_models_path}/DQN_{pop_i}.pt")
        accelerator.wait_for_everyone()
        for model in pop:
            model.wrap_models()

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
