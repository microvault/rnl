import numpy as np
import torch
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.probe_envs import (
    ConstantRewardContActionsEnv,
    DiscountedRewardContActionsEnv,
    FixedObsPolicyContActionsEnv,
    ObsDependentRewardContActionsEnv,
    PolicyContActionsEnv,
    check_policy_on_policy_with_probe_env,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.dqn.policies import DQNPolicy

import wandb
from rnl.algorithms.ppo import PPO as agent_algo
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
from rnl.training.utils import create_population, make_vect_envs


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
    probe: bool,
):

    NET_CONFIG = {
        "arch": network_config.arch,
        "hidden_size": network_config.hidden_size,
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
                "lr": 0.001,
            }

            check_policy_on_policy_with_probe_env(
                env, agent_algo, algo_args, learn_steps, device="cpu"
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


def learn_with_sb3(
    trainer_config: TrainerConfig,
    network_config: NetworkConfig,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
):

    config_dict = {
        "Trainer Config": trainer_config.__dict__,
        "Network Config": network_config.__dict__,
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
    run = wandb.init(
        project="rnl",
        config=config_dict,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
    )

    env = NaviEnv(
            robot_config, sensor_config, env_config, render_config, use_render=False
        )
    print("\nCheck environment ...")
    check_env(env)

    policy_kwargs_on_policy = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(
            pi=[network_config.hidden_size[0], network_config.hidden_size[1]],
            vf=[network_config.hidden_size[0], network_config.hidden_size[1]],
        ),
    )

    def make_env():
        env = NaviEnv(
            robot_config, sensor_config, env_config, render_config, use_render=False
        )
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    if trainer_config.algorithms == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            batch_size=trainer_config.batch_size,
            verbose=1,
            policy_kwargs=policy_kwargs_on_policy,
            tensorboard_log=f"runs/{run.id}",
            device=trainer_config.device,
            seed=trainer_config.seed,
        )

        print("\nInitiate PPO training ...")

    elif trainer_config.algorithms == "A2C":
        model = A2C("MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs_on_policy,
            tensorboard_log=f"runs/{run.id}",
            device=trainer_config.device,
            seed=trainer_config.seed,
        )
        print("\nInitiate A2C training ...")


    elif trainer_config.algorithms == "DQN":
        model = DQN(
            DQNPolicy,
            env,
            batch_size=trainer_config.batch_size,
            buffer_size=trainer_config.buffer_size,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device=trainer_config.device,
            seed=trainer_config.seed,
        )

        print("\nInitiate DQN training ...")

    else:
        print("Invalid algorithm")

    model.learn(
        total_timesteps=trainer_config.max_timestep_global,
        callback=WandbCallback(
            gradient_save_freq=0,
            model_save_path=f"models_{trainer_config.algorithms}/{run.id}",
            verbose=2,
        ),
    )
    run.finish()

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
    ep_rewards = np.zeros(num_envs)
    ep_lengths = np.zeros(num_envs)

    completed_rewards = []
    completed_lengths = []

    for i in pbar:
        # Ação: exemplo usando ações aleatórias
        actions = env.action_space.sample()

        # Passo no ambiente
        next_state, rewards, terminated, truncated, info = env.step(actions)
        steps += 1

        # Atualiza recompensas e comprimentos
        ep_rewards += np.array(rewards)
        ep_lengths += 1  # Incrementa o comprimento do episódio

        # Identifica quais ambientes terminaram neste passo
        done = np.logical_or(terminated, truncated)
        done_indices = np.where(done)[0]

        if done_indices.size > 0:
            for idx in done_indices:
                # Adiciona as recompensas e comprimentos dos episódios concluídos às listas
                completed_rewards.append(ep_rewards[idx])
                completed_lengths.append(ep_lengths[idx])

                # Reseta as recompensas e comprimentos para os ambientes que terminaram
                ep_rewards[idx] = 0
                ep_lengths[idx] = 0

        # Atualiza a barra de progresso com informações relevantes
        if len(completed_rewards) > 0 and len(completed_lengths) > 0:
            avg_reward = np.mean(
                completed_rewards[-100:]
            )  # Média dos últimos 100 episódios
            avg_length = np.mean(
                completed_lengths[-100:]
            )  # Média dos últimos 100 episódios
        else:
            avg_reward = 0
            avg_length = 0

        pbar.set_postfix(
            {
                "Episódios Completos": len(completed_rewards),
                "Recompensa Média (últ 100)": f"{avg_reward:.2f}",
                "Comprimento Médio (últ 100)": f"{avg_length:.2f}",
            }
        )

        # Opcional: Parar se todos os ambientes completaram um episódio (ajuste conforme necessário)
        # if len(completed_rewards) >= desired_number_of_episodes:
        #     break

    # Após o loop, é possível que alguns ambientes ainda estejam ativos e não tenham terminado
    # Você pode optar por coletar esses dados também
    for idx in range(num_envs):
        if ep_lengths[idx] > 0:
            completed_rewards.append(ep_rewards[idx])
            completed_lengths.append(ep_lengths[idx])

    env.close()

    # Converter as listas para arrays numpy para facilitar a análise posterior
    completed_rewards = np.array(completed_rewards)
    completed_lengths = np.array(completed_lengths)

    # obstacles_scores = []
    # collision_scores = []
    # orientation_scores = []
    # progress_scores = []
    # time_scores = []
    # rewards = []

    # # Ler o arquivo CSV
    # with open(csv_file, mode="r") as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         obstacles_scores.append(float(row[0]))
    #         collision_scores.append(float(row[1]))
    #         orientation_scores.append(float(row[2]))
    #         progress_scores.append(float(row[3]))
    #         time_scores.append(float(row[4]))
    #         rewards.append(float(row[5]))

    # # Gerar o eixo X como o índice das linhas
    # steps = list(range(1, len(rewards) + 1))

    # # Definir uma lista de componentes para facilitar a iteração
    # components = [
    #     ("Obstacles Score", obstacles_scores, "brown"),
    #     ("Collision Score", collision_scores, "red"),
    #     ("Orientation Score", orientation_scores, "green"),
    #     ("Progress Score", progress_scores, "blue"),
    #     ("Time Score", time_scores, "orange"),
    #     ("Total Reward", rewards, "purple"),
    #     ("ep_len_mean", completed_lengths, "gray"),
    #     ("ep_rew_mean", completed_rewards, "black"),
    # ]

    # # Configurar o layout da figura com subplots
    # num_plots = len(components)
    # cols = 2
    # rows = (num_plots + cols - 1) // cols  # Calcula o número de linhas necessárias

    # plt.figure(figsize=(10, 5 * rows))  # Ajusta a altura com base no número de linhas

    # for idx, (title, data, color) in enumerate(components, 1):
    #     ax = plt.subplot(rows, cols, idx)
    #     ax.plot(steps, data, label=title, color=color, linestyle="-", linewidth=1.5)
    #     ax.set_ylabel(title, fontsize=14)
    #     ax.legend(fontsize=12)
    #     ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    #     ax.tick_params(axis="x", labelsize=12)
    #     ax.tick_params(axis="y", labelsize=12)

    #     # Calcular estatísticas
    #     mean_val = np.mean(data)
    #     min_val = np.min(data)
    #     max_val = np.max(data)

    #     # Adicionar texto abaixo do plot com as estatísticas
    #     # A posição (0.5, -0.25) coloca o texto centralizado abaixo do gráfico
    #     ax.text(
    #         0.5,
    #         -0.25,
    #         f"Média: {mean_val:.2f} | Mínimo: {min_val:.2f} | Máximo: {max_val:.2f}",
    #         transform=ax.transAxes,
    #         ha="center",
    #         fontsize=12,
    #     )

    # # Ajustar layout para evitar sobreposição
    # plt.tight_layout()

    # # Ajustar espaço adicional na parte inferior para os textos
    # plt.subplots_adjust(bottom=0.15)

    # # Exibir o gráfico
    # plt.show()
