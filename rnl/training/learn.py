import csv

import matplotlib.pyplot as plt
import numpy as np
from rnl.algorithms.ppo import PPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.probe_envs import (
    ConstantRewardContActionsEnv,
    DiscountedRewardContActionsEnv,
    FixedObsPolicyContActionsEnv,
    ObsDependentRewardContActionsEnv,
    PolicyContActionsEnv,
    check_policy_on_policy_with_probe_env
)
from tqdm import trange

from rnl.configs.config import (
    EnvConfig,
    HPOConfig,
    NetworkConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.environment.env import NaviEnv
from rnl.training.policy import train_on_policy
from rnl.training.utils import (
    create_population,
    make_vect_envs,
)


def training(
    trainer_config: TrainerConfig,
    hpo_config: HPOConfig,
    network_config: NetworkConfig,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    pretrained_model: bool,
    train_docker: bool,
    debug: bool,
    probe: bool,
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
        "DISCRETE_ACTIONS": True,
        "BATCH_SIZE": trainer_config.batch_size,
        "LR": trainer_config.lr,
        "LEARN_STEP": trainer_config.learn_step,
        "GAMMA": trainer_config.gamma,
        "GAE_LAMBDA": trainer_config.gae_lambda,
        "ACTION_STD_INIT": trainer_config.action_std_init,
        "CLIP_COEF": trainer_config.clip_coef,
        "ENT_COEF": trainer_config.ent_coef,
        "VF_COEF": trainer_config.vf_coef,
        "MAX_GRAD_NORM": trainer_config.max_grad_norm,
        "UPDATE_EPOCHS": trainer_config.update_epochs,
        "TARGET_KL": None,
        "CHANNELS_LAST": False,
    }

    MUTATION_PARAMS = {
        "NO_MUT": hpo_config.no_mutation,  # No mutation
        "ARCH_MUT": hpo_config.arch_mutation,  # Architecture mutation
        "NEW_LAYER": hpo_config.new_layer,  # New layer mutation
        "PARAMS_MUT": hpo_config.param_mutation,  # Network parameters mutation
        "ACT_MUT": hpo_config.active_mutation,  # Activation layer mutation
        "RL_HP_MUT": hpo_config.hp_mutation,  # Learning HP mutation
        "RL_HP_SELECTION": hpo_config.hp_mutation_selection,  # Learning HPs to choose from
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
    max_steps = trainer_config.max_timestep_global
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
        use_render=False,
    )
    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    env_navigation = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=True,
    )

    pop = create_population(
        env_navigation=env_navigation,
        algo="PPO",
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=False,
        net_config=NET_CONFIG,
        INIT_HP=HYPERPARAM,
        population_size=hpo_config.population_size,
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
        "Trainer Config": trainer_config.__dict__,
        "HPO Config": hpo_config.__dict__,
        "Network Config": network_config.__dict__,
    }

    for config_name, config_values in config_dict.items():
        print(f"\n#------ {config_name} ----#")
        max_key_length = max(len(key) for key in config_values.keys())
        for key, value in config_values.items():
            print(f"{key.ljust(max_key_length)} : {value}")

    print()
    if probe:
        print("Probing environments")
        cont_vector_envs = [
            (ConstantRewardContActionsEnv(), 1000),
            (ObsDependentRewardContActionsEnv(), 1000),
            (DiscountedRewardContActionsEnv(), 5000),
            (FixedObsPolicyContActionsEnv(), 3000),
            (PolicyContActionsEnv(), 3000),
        ]

        for env, learn_steps in cont_vector_envs:
            algo_args = {
                "state_dim": (env.observation_space.n,),
                "action_dim": env.action_space.shape[0],
                "one_hot": True if env.observation_space.n > 1 else False,
                "discrete_actions": False,
                "lr": 0.001
            }

            check_policy_on_policy_with_probe_env(
                env, PPO, algo_args, learn_steps, device="cpu"
            )

    if train_docker:
        print("Training in Docker")
        trained_pop, pop_fitnesses = train_on_policy(
            env=env,
            env_name="rnl",
            algo="PPO",
            pop=pop,
            INIT_HP=HYPERPARAM,
            MUT_P=MUTATION_PARAMS,
            swap_channels=False,
            max_steps=max_steps,
            evo_steps=evo_steps,
            eval_steps=eval_steps,
            eval_loop=eval_loop,
            target=None,
            tournament=tournament,
            mutation=mutations,
            checkpoint=trainer_config.checkpoint,
            checkpoint_path=trainer_config.checkpoint_path,
            overwrite_checkpoints=trainer_config.overwrite_checkpoints,
            save_elite=hpo_config.save_elite,
            elite_path=hpo_config.elite_path,
            wb=trainer_config.use_wandb,
            verbose=True,
            accelerator=None,
            wandb_api_key=trainer_config.wandb_api_key,
        )

    else:
        print("Training locally")
        pbar = trange(max_steps, unit="step")
        while np.less([agent.steps[-1] for agent in pop], max_steps).all():
            pop_episode_scores = []
            for agent in pop:  # Loop through population
                state, info = env.reset()  # Reset environment at start of episode
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
                        next_state, reward, terminated, truncated, info = env.step(
                            action
                        )

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
        use_render=True,
    )
    print("\n#------ Info Env ----#")
    obs_space = env.observation_space
    state_dim = obs_space.shape
    print("States dim: ", state_dim)

    action_space = env.action_space
    action_dim = action_space.n
    print("Action dim:", action_dim)

    env.reset()
    env.render()


def probe_envs(
    csv_file: str,
    num_envs: int,
    max_steps: int,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    pretrained_model: bool,
):

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

    print()
    pbar = trange(max_steps, desc="Probe envs", unit="step")

    env = make_vect_envs(
        num_envs=num_envs,
        robot_config=robot_config,
        sensor_config=sensor_config,
        env_config=env_config,
        render_config=render_config,
        use_render=False,
    )

    state, info = env.reset()
    scores = np.zeros(num_envs)
    steps = 0

    for i in pbar:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        steps += 1
        scores += np.array(reward)

        if np.any(terminated) or np.any(truncated) or i == max_steps - 1:
            state, info = env.reset()
            scores = np.zeros(num_envs)

    env.close()

    obstacles_scores = []
    collision_scores = []
    orientation_scores = []
    progress_scores = []
    time_scores = []
    rewards = []

    # Ler o arquivo CSV
    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            obstacles_scores.append(float(row[0]))
            collision_scores.append(float(row[1]))
            orientation_scores.append(float(row[2]))
            progress_scores.append(float(row[3]))
            time_scores.append(float(row[4]))
            rewards.append(float(row[5]))

    # Gerar o eixo X como o índice das linhas
    steps = list(range(1, len(rewards) + 1))

    # Definir uma lista de componentes para facilitar a iteração
    components = [
        ("Obstacles Score", obstacles_scores, "brown"),
        ("Collision Score", collision_scores, "red"),
        ("Orientation Score", orientation_scores, "green"),
        ("Progress Score", progress_scores, "blue"),
        ("Time Score", time_scores, "orange"),
        ("Total Reward", rewards, "purple"),
    ]

    # Configurar o layout da figura com subplots
    num_plots = len(components)
    cols = 2
    rows = (num_plots + cols - 1) // cols  # Calcula o número de linhas necessárias

    plt.figure(figsize=(10, 5 * rows))  # Ajusta a altura com base no número de linhas

    for idx, (title, data, color) in enumerate(components, 1):
        ax = plt.subplot(rows, cols, idx)
        ax.plot(steps, data, label=title, color=color, linestyle="-", linewidth=1.5)
        ax.set_ylabel(title, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        # Calcular estatísticas
        mean_val = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)

        # Adicionar texto abaixo do plot com as estatísticas
        # A posição (0.5, -0.25) coloca o texto centralizado abaixo do gráfico
        ax.text(
            0.5,
            -0.25,
            f"Média: {mean_val:.2f} | Mínimo: {min_val:.2f} | Máximo: {max_val:.2f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
        )

    # Ajustar layout para evitar sobreposição
    plt.tight_layout()

    # Ajustar espaço adicional na parte inferior para os textos
    plt.subplots_adjust(bottom=0.15)

    # Exibir o gráfico
    plt.show()
