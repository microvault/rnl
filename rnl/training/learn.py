import numpy as np
import torch
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from tqdm import trange

from rnl.configs.config import (
    AgentConfig,
    EnvConfig,
    HPOConfig,
    NetworkConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.environment.environment_navigation import NaviEnv
from rnl.training.utils import make_vect_envs


def training(
    agent_config: AgentConfig,
    trainer_config: TrainerConfig,
    hpo_config: HPOConfig,
    network_config: NetworkConfig,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    pretrained_model: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_dict = {
        "Agent Config": agent_config.__dict__,
        "Trainer Config": trainer_config.__dict__,
        "HPO Config": hpo_config.__dict__,
        "Network Config": network_config.__dict__,
    }

    for config_name, config_values in config_dict.items():
        print(f"\n#------ {config_name} ----#")
        max_key_length = max(len(key) for key in config_values.keys())
        for key, value in config_values.items():
            print(f"{key.ljust(max_key_length)} : {value}")

    INIT_PARAM = {
        "ENV_NAME": "rnl",  # Gym environment name
        "ALGO": "Rainbow DQN",  # Algorithm
        "DOUBLE": True,  # Use double Q-learning
        "CHANNELS_LAST": False,  # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "BATCH_SIZE": trainer_config.batch_size,  # Batch size
        "LR": trainer_config.lr,  # Learning rate
        "MAX_STEPS": trainer_config.max_steps,  # Max no. steps
        "TARGET_SCORE": trainer_config.target_score,  # Early training stop at avg score of last 100 episodes
        "GAMMA": agent_config.gamma,  # Discount factor
        "TAU": agent_config.tau,  # For soft update of target parameters
        "BETA": agent_config.beta,  # PER beta
        "PRIOR_EPS": agent_config.prior_eps,  # PER epsilon
        "MEMORY_SIZE": agent_config.memory_size,  # Max memory buffer size
        "LEARN_STEP": trainer_config.learn_step,  # Learning frequency
        "TOURN_SIZE": hpo_config.tourn_size,  # Tournament size
        "ELITISM": hpo_config.elitism,  # Elitism in tournament selection
        "POP_SIZE": hpo_config.population_size,  # Population size
        "EVO_STEPS": hpo_config.evo_steps,  # Evolution frequency
        "EVAL_STEPS": hpo_config.eval_steps,  # Evaluation steps
        "EVAL_LOOP": hpo_config.eval_loop,  # Evaluation episodes
        "LEARNING_DELAY": trainer_config.learning_delay,  # Steps before starting learning
        "WANDB": trainer_config.use_wandb,  # Log with Weights and Biases
        "CHECKPOINT": trainer_config.checkpoint,  # Checkpoint frequency
        "CHECKPOINT_PATH": trainer_config.checkpoint_path,  # Checkpoint path
        "SAVE_ELITE": hpo_config.save_elite,  # Save elite agent
        "ELITE_PATH": hpo_config.elite_path,  # Elite agent path
        "ACCELERATOR": None,
        "VERBOSE": True,
        "TORCH_COMPILER": True,
        "EPS_START": trainer_config.eps_start,
        "EPS_END": trainer_config.eps_end,
        "EPS_DECAY": trainer_config.eps_decay,
        "N_STEP": agent_config.n_step,
        "PER": agent_config.per,
        "N_STEP_MEMORY": trainer_config.n_step_memory,
        "NUM_ATOMS": agent_config.num_atoms,
        "V_MIN": agent_config.v_min,
        "V_MAX": agent_config.v_max,
    }

    NET_CONFIG = {
        "arch": network_config.arch,  # Network architecture
        "hidden_size": network_config.hidden_size,  # Actor hidden size
        "mlp_activation": network_config.mlp_activation,
        "mlp_output_activation": network_config.mlp_output_activation,
        "min_hidden_layers": network_config.min_hidden_layers,
        "max_hidden_layers": network_config.max_hidden_layers,
        "min_mlp_nodes": network_config.min_mlp_nodes,
        "max_mlp_nodes": network_config.max_mlp_nodes,
    }

    MUTATION_PARAMS = {
        "NO_MUT": hpo_config.no_mutation,  # No mutation
        "ARCH_MUT": hpo_config.arch_mutation,  # Architecture mutation
        "NEW_LAYER": hpo_config.new_layer,  # New layer mutation
        "PARAMS_MUT": hpo_config.param_mutation,  # Network parameters mutation
        "ACT_MUT": hpo_config.active_mutation,  # Activation layer mutation
        "RL_HP_MUT": hpo_config.hp_mutation,  # Learning HP mutation
        "RL_HP_SELECTION": hpo_config.hp_mutation_selection,
        "MUT_SD": hpo_config.mutation_strength,  # Mutation strength
        "MIN_LR": hpo_config.min_lr,
        "MAX_LR": hpo_config.max_lr,
        "MIN_LEARN_STEP": hpo_config.min_learn_step,
        "MAX_LEARN_STEP": hpo_config.max_learn_step,
        "MIN_BATCH_SIZE": hpo_config.min_batch_size,
        "MAX_BATCH_SIZE": hpo_config.max_batch_size,
        "AGENTS_IDS": None,
        "MUTATE_ELITE": hpo_config.mutate_elite,
        "RAND_SEED": hpo_config.rand_seed,  # Random seed
        "ACTIVATION": hpo_config.activation,  # Activation functions to choose from
    }

    num_envs = trainer_config.num_envs

    env = make_vect_envs(
        num_envs=num_envs,
        robot_config=robot_config,
        sensor_config=sensor_config,
        env_config=env_config,
        render_config=render_config,
    )
    one_hot = False
    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    pop = create_population(
        algo=INIT_PARAM["ALGO"],  # Algorithm
        state_dim=state_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=one_hot,  # One-hot encoding
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_PARAM,  # Initial hyperparameters
        population_size=INIT_PARAM["POP_SIZE"],  # Population size
        num_envs=num_envs,  # Number of vectorized environments
        device=device,
        accelerator=INIT_PARAM["ACCELERATOR"],
        torch_compiler=INIT_PARAM["TORCH_COMPILER"],
    )

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=INIT_PARAM["MEMORY_SIZE"],  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )

    tournament = TournamentSelection(
        tournament_size=INIT_PARAM["TOURN_SIZE"],  # Tournament selection size
        elitism=INIT_PARAM["ELITISM"],  # Elitism in tournament selection
        population_size=INIT_PARAM["POP_SIZE"],  # Population size
        eval_loop=INIT_PARAM["EVAL_LOOP"],  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        algo=INIT_PARAM["ALGO"],  # Algorithm
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
        activation_selection=MUTATION_PARAMS[
            "ACTIVATION"
        ],  # Activation functions to choose from
        min_lr=MUTATION_PARAMS["MIN_LR"],
        max_lr=MUTATION_PARAMS["MAX_LR"],
        min_learn_step=MUTATION_PARAMS["MIN_LEARN_STEP"],
        max_learn_step=MUTATION_PARAMS["MAX_LEARN_STEP"],
        min_batch_size=MUTATION_PARAMS["MIN_BATCH_SIZE"],
        max_batch_size=MUTATION_PARAMS["MAX_BATCH_SIZE"],
        agent_ids=MUTATION_PARAMS["AGENTS_IDS"],
        mutate_elite=MUTATION_PARAMS["MUTATE_ELITE"],
        arch=NET_CONFIG["arch"],  # Network architecture
        rand_seed=MUTATION_PARAMS["RAND_SEED"],  # Random seed
        device=device,
        accelerator=INIT_PARAM["ACCELERATOR"],
    )

    eps_start = INIT_PARAM["EPS_START"]
    evo_steps = INIT_PARAM["EVO_STEPS"]
    eps_end = INIT_PARAM["EPS_END"]
    eps_decay = INIT_PARAM["EPS_DECAY"]
    epsilon = eps_start
    learning_delay = INIT_PARAM["LEARNING_DELAY"]
    total_steps = 0
    eval_steps = INIT_PARAM["EVAL_STEPS"]
    eval_loop = INIT_PARAM["EVAL_LOOP"]
    max_steps = INIT_PARAM["MAX_STEPS"]
    # TRAINING LOOP
    print("Training...")
    pbar = trange(INIT_PARAM["MAX_STEPS"], unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            epsilon = eps_start

            for idx_step in range(evo_steps // num_envs):
                action = agent.get_action(state, epsilon)  # Get next action from agent
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
                memory.save_to_memory(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                    is_vectorised=True,
                )

                # Learn according to learning frequency
                if memory.counter > learning_delay and len(memory) >= agent.batch_size:
                    for _ in range(num_envs // agent.learn_step):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

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

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Scores: {mean_scores}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(
            f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()


def inference(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    pretrained_model=False,
):

    text = [
        r"+--------------------+",
        r" ____  _   _ _",
        r"|  _ \| \ | | |",
        r"| |_) |  \| | |",
        r"|  _ <| |\  | |___",
        r"|_| \_\_| \_|_____|",
        r"+--------------------+",
    ]

    for line in text:
        print(line)

    config_dict = {
        "Robot Config": robot_config.__dict__,
        "Sensor Config": sensor_config.__dict__,
        "Env Config": env_config.__dict__,
        "Render Config": render_config.__dict__,
    }

    for config_name, config_values in config_dict.items():
        print(f"\n#------ {config_name} ----#")
        max_key_length = max(len(key) for key in config_values.keys())
        for key, value in config_values.items():
            print(f"{key.ljust(max_key_length)} : {value}")

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        pretrained_model=False,
        use_render=True,
    )

    env.reset()
    env.render()
