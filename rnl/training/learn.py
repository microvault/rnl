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
    pretrained_model: bool,
    wb: bool,
    api_key: str,
    checkpoint_path: str,
    checkpoint: int,
    overwrite_checkpoints: bool,
):

    env = make_vect_envs(
        num_envs=trainer_config.num_envs,
        robot_config=robot_config,
        sensor_config=sensor_config,
        env_config=env_config,
        render_config=render_config,
        pretrained_model=pretrained_model,
    )

    net_config = {
        "hidden_size": network_config.hidden_size,
        "mlp_activation": network_config.mlp_activation,
        "mlp_output_activation": network_config.mlp_output_activation,
        "min_hidden_layers": network_config.min_hidden_layers,
        "max_hidden_layers": network_config.max_hidden_layers,
        "min_mlp_nodes": network_config.min_mlp_nodes,
        "max_mlp_nodes": network_config.max_mlp_nodes,
    }

    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    field_names = ["state", "action", "reward", "next_state", "termination"]
    memory = PrioritizedReplayBuffer(
        memory_size=agent_config.memory_size,
        field_names=field_names,
        num_envs=trainer_config.num_envs,
        alpha=agent_config.alpha,
        n_step=agent_config.n_step,
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
        state_dim=int(state_dim[0]),  # State dimension
        action_dim=action_dim,  # Action dimension
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
        no_mutation=hpo_config.no_mutation,  # No mutation
        architecture=hpo_config.arch_mutation,  # Architecture mutation
        new_layer_prob=hpo_config.new_layer,  # New layer mutation
        parameters=hpo_config.param_mutation,  # Network parameters mutation
        activation=hpo_config.active_mutation,  # Activation layer mutation
        rl_hp=hpo_config.hp_mutation,  # Learning HP mutation
        rl_hp_selection=hpo_config.hp_mutation_selection,  # Learning HPs to choose from
        mutation_sd=hpo_config.mutation_strength,  # Mutation strength
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
        config=config_dict,
        env=env,
        pop=agent_pop,
        memory=memory,
        n_step_memory=n_step_memory,
        max_steps=trainer_config.max_steps,
        evo_steps=hpo_config.evolution_steps,
        eval_steps=trainer_config.evaluation_steps,
        eval_loop=trainer_config.evaluation_loop,
        learning_delay=trainer_config.learning_delay,
        target=trainer_config.target_score,
        eps_start=agent_config.epsilon_start,
        eps_end=agent_config.epsilon_end,
        eps_decay=agent_config.epsilon_decay,
        tournament=tournament,
        mutation=mutations,
        wb=wb,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        overwrite_checkpoints=overwrite_checkpoints,
        wandb_api_key=api_key,
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
        robot_config, sensor_config, env_config, render_config, pretrained_model=False
    )

    env.reset()
    env.render()
