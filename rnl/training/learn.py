import numpy as np
import torch
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
from rnl.environment.env import NaviEnv
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

    HYPERPARAM = {
        "POP_SIZE": 6,  # Population size
        "DISCRETE_ACTIONS": True,  # Discrete action space
        "BATCH_SIZE": 128,  # Batch size
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
    }

    num_envs = trainer_config.num_envs
    max_steps = trainer_config.max_steps
    evo_steps = hpo_config.evo_steps
    eval_steps = hpo_config.eval_steps
    eval_loop = hpo_config.eval_loop
    total_steps = 0

    env = make_vect_envs(
        num_envs=num_envs,
        robot_config=robot_config,
        sensor_config=sensor_config,
        env_config=env_config,
        render_config=render_config,
    )
    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    pop = create_population(
        algo="PPO",
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=False,
        net_config=NET_CONFIG,
        INIT_HP=HYPERPARAM,
        population_size=hpo_config.tourn_size,
        num_envs=num_envs,
        device=trainer_config.device,
    )

    tournament = TournamentSelection(
        tournament_size=hpo_config.tourn_size,
        elitism=hpo_config.elitism,
        population_size=hpo_config.population_size,
        eval_loop=hpo_config.eval_loop,
    )

    mutations = Mutations(
        algo="PPO",
        no_mutation=hpo_config.no_mutation,
        architecture=hpo_config.arch_mutation,
        new_layer_prob=hpo_config.new_layer,
        parameters=hpo_config.param_mutation,
        activation=hpo_config.active_mutation,
        rl_hp=hpo_config.hp_mutation,
        rl_hp_selection=hpo_config.hp_mutation_selection,
        mutation_sd=hpo_config.mutation_strength,
        activation_selection=hpo_config.activation,
        min_lr=hpo_config.min_lr,
        max_lr=hpo_config.max_lr,
        min_learn_step=hpo_config.min_learn_step,
        max_learn_step=hpo_config.max_learn_step,
        min_batch_size=hpo_config.min_batch_size,
        max_batch_size=hpo_config.max_batch_size,
        agent_ids=None,
        mutate_elite=hpo_config.mutate_elite,
        arch="mlp",
        rand_seed=hpo_config.rand_seed,
        device=trainer_config.device,
        accelerator=None,
    )

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

    pbar = trange(max_steps, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:
            state, info = env.reset()
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0

            for _ in range(-(evo_steps // -agent.learn_step)):

                states = []
                actions = []
                log_probs = []
                rewards = []
                dones = []
                values = []

                learn_steps = 0

                for idx_step in range(-(agent.learn_step // -num_envs)):
                    # Get next action from agent
                    action, log_prob, _, value = agent.get_action(state)

                    # Act in environment
                    next_state, reward, terminated, truncated, info = env.step(action)

                    total_steps += num_envs
                    steps += num_envs
                    learn_steps += num_envs

                    states.append(state)
                    actions.append(action)
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    dones.append(terminated)
                    values.append(value)

                    state = next_state
                    scores += np.array(reward)

                    for idx, (d, t) in enumerate(zip(terminated, truncated)):
                        if d or t:
                            completed_episode_scores.append(scores[idx])
                            agent.scores.append(scores[idx])
                            scores[idx] = 0

                pbar.update(learn_steps // len(pop))

                experiences = (
                    states,
                    actions,
                    log_probs,
                    rewards,
                    dones,
                    values,
                    next_state,
                )
                # Learn according to agent's RL algorithm
                agent.learn(experiences)

            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=False,
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
    pretrained_model: bool,
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
        pretrained_model,
        use_render=True,
    )
    print("\n#------ Info Env ----#")
    state_dim = env.observation_space
    action_dim = env.action_space.shape
    print(f"{state_dim} : state dim")
    print(f"{action_dim} : action dim")

    env.reset()
    env.render()
