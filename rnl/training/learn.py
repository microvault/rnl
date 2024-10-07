from rnl.components.replay_buffer import MultiStepReplayBuffer, PrioritizedReplayBuffer
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
from rnl.hpo.mutation import Mutations
from rnl.hpo.tournament import TournamentSelection
from rnl.training.train_off_policy import train_off_policy
from rnl.training.utils import create_population, make_vect_envs


def training(
    agent_config: AgentConfig,
    trainer_config: TrainerConfig,
    hpo_config: HPOConfig,
    network_config: NetworkConfig,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    pretrained_model,
):

    env = make_vect_envs(
        env_name=NaviEnv,
        num_envs=trainer_config.num_envs,
        robot_config=robot_config,
        sensor_config=sensor_config,
        env_config=env_config,
        render_config=render_config,
    )

    net_config = {
        "arch": "mlp",
        "hidden_size": network_config.hidden_size,  # Network hidden size
    }

    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    field_names = ["state", "action", "reward", "next_state", "termination"]
    memory = PrioritizedReplayBuffer(
        memory_size=agent_config.memory_size,
        field_names=field_names,
        num_envs=trainer_config.num_envs,
        alpha=agent_config.alpha,
        gamma=agent_config.gamma,
        device=trainer_config.device,
    )

    n_step_memory = MultiStepReplayBuffer(
        memory_size=agent_config.memory_size,
        field_names=field_names,
        num_envs=trainer_config.num_envs,
        n_step=agent_config.n_step,
        gamma=agent_config.gamma,
        device=trainer_config.device,
    )

    agent_pop = create_population(
        algo="RainbowDQN",  # Algorithm
        state_dim=state_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=False,  # One-hot encoding
        net_config=net_config,  # Network configuration
        agent_config=agent_config,  # Agent configuration
        trainer_config=trainer_config,  # Trainer configuration
        population_size=hpo_config.population_size,  # Population size
        num_envs=trainer_config.num_envs,  # Number of vectorized environments
        device=trainer_config.device,
    )

    tournament = TournamentSelection(
        tournament_size=hpo_config.tourn_size,  # Tournament selection size
        elitism=hpo_config.elitism,  # Elitism in tournament selection
        population_size=hpo_config.population_size,  # Population size
        eval_loop=trainer_config.evaluation_loop,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        algo="Rainbow DQN",  # Algorithm
        no_mutation=hpo_config.no_mutation,  # No mutation
        architecture=hpo_config.arch_mutation,  # Architecture mutation
        new_layer_prob=hpo_config.new_layer,  # New layer mutation
        parameters=hpo_config.param_mutation,  # Network parameters mutation
        activation=hpo_config.active_mutation,  # Activation layer mutation
        rl_hp=hpo_config.hp_mutation,  # Learning HP mutation
        rl_hp_selection=hpo_config.hp_mutation_selection,  # Learning HPs to choose from
        mutation_sd=hpo_config.mutation_strength,  # Mutation strength
        arch="mlp",  # Network architecture
        rand_seed=trainer_config.seed,  # Random seed
        device=trainer_config.device,
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

    trained_pop, pop_fitnesses = train_off_policy(
        env=env,
        env_name="NaviEnv-v0",
        algo="Rainbow DQN",
        pop=agent_pop,
        memory=memory,
        n_step_memory=n_step_memory,
        max_steps=trainer_config.max_steps,
        evo_steps=hpo_config.evolution_steps,
        eval_steps=trainer_config.evaluation_steps,
        eval_loop=trainer_config.evaluation_loop,
        learning_delay=trainer_config.learning_delay,
        target=trainer_config.target_score,
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
        r"_____________________",
    ]

    for line in text:
        print(line)

    env = NaviEnv(robot_config, sensor_config, env_config, render_config)

    env.reset()
    env.render()
