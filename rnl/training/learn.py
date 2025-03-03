import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from agilerl.utils.utils import create_population
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from torch import nn
from tqdm import trange
from wandb.integration.sb3 import WandbCallback

from rnl.configs.actions import FastActions, SlowActions
from rnl.configs.config import (
    EnvConfig,
    NetworkConfig,
    ProbeEnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.engine.utils import set_seed
from rnl.engine.vector import make_vect_envs
from rnl.environment.env import NaviEnv


def training(
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

    table_width = 45
    horizontal_line = "-" * table_width
    print(horizontal_line)

    for config_name, config_values in config_dict.items():
        print(f"| {config_name + '/':<41} |")
        print(horizontal_line)
        for key, value in config_values.items():
            print(f"|    {key.ljust(20)} | {str(value).ljust(15)} |")
        print(horizontal_line)

    if trainer_config.use_wandb:
        run = wandb.init(
            name=trainer_config.checkpoint,
            project=trainer_config.name,
            config=config_dict,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=False,
        )

    slow_actions = SlowActions()
    fast_actions = FastActions()

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=False,
        actions_cfg=fast_actions,
    )

    print("\nCheck environment ...")
    check_env(env)

    activation_fn_map = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
    }
    activation_fn = activation_fn_map[network_config.mlp_activation]

    policy_kwargs_on_policy = dict(
        activation_fn=activation_fn,
        net_arch=dict(
            pi=[network_config.hidden_size[0], network_config.hidden_size[1]],
            vf=[network_config.hidden_size[0], network_config.hidden_size[1]],
        ),
    )

    def make_env():
        slow_actions = SlowActions()
        fast_actions = FastActions()

        env = NaviEnv(
            robot_config,
            sensor_config,
            env_config,
            render_config,
            use_render=False,
            actions_cfg=fast_actions,
        )
        env = Monitor(env)
        return env

    # Parallel environments
    vec_env = make_vec_env(make_env, n_envs=trainer_config.num_envs)
    if trainer_config.use_wandb:
        model = PPO(
            "MlpPolicy",
            vec_env,
            batch_size=trainer_config.batch_size,
            verbose=1,
            learning_rate=trainer_config.lr,
            policy_kwargs=policy_kwargs_on_policy,
            n_steps=trainer_config.learn_step,
            vf_coef=trainer_config.vf_coef,
            ent_coef=trainer_config.ent_coef,
            device=trainer_config.device,
            tensorboard_log=f"runs/{run.id}",
            max_grad_norm=trainer_config.max_grad_norm,
            n_epochs=trainer_config.update_epochs,
            seed=trainer_config.seed,
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            batch_size=trainer_config.batch_size,
            verbose=1,
            learning_rate=trainer_config.lr,
            policy_kwargs=policy_kwargs_on_policy,
            n_steps=trainer_config.learn_step,
            vf_coef=trainer_config.vf_coef,
            ent_coef=trainer_config.ent_coef,
            device=trainer_config.device,
            max_grad_norm=trainer_config.max_grad_norm,
            n_epochs=trainer_config.update_epochs,
            seed=trainer_config.seed,
        )

    print("\nInitiate PPO training ...")

    if trainer_config.use_wandb:
        model.learn(
            total_timesteps=trainer_config.max_timestep_global,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"model_{trainer_config.checkpoint}/{run.id}",
                verbose=2,
            ),
        )
        run.finish()

    else:
        model.learn(total_timesteps=trainer_config.max_timestep_global)


def inference(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
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

    table_width = 45
    horizontal_line = "-" * table_width
    print(horizontal_line)

    for config_name, config_values in config_dict.items():
        print(f"| {config_name + '/':<41} |")
        print(horizontal_line)
        for key, value in config_values.items():
            print(f"|    {key.ljust(20)} | {str(value).ljust(15)} |")
        print(horizontal_line)

    slow_actions = SlowActions()
    fast_actions = FastActions()

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=True,
        actions_cfg=slow_actions,
    )

    env.reset()
    env.render()


def probe_envs(
    num_envs, max_steps, robot_config, sensor_config, env_config, render_config
):

    # set_seed(6)  # Define a semente

    # Exibe as configurações
    probe_config = ProbeEnvConfig(num_envs=num_envs, max_steps=max_steps)
    config_dict = {
        "Robot Config": robot_config.__dict__,
        "Sensor Config": sensor_config.__dict__,
        "Env Config": env_config.__dict__,
        "Render Config": render_config.__dict__,
        "Probe Config": probe_config.__dict__,
    }
    table_width = 45
    horizontal_line = "-" * table_width
    print(horizontal_line)
    for name, cfg in config_dict.items():
        print(f"| {name + '/':<41} |")
        print(horizontal_line)
        for key, value in cfg.items():
            print(f"|    {key.ljust(20)} | {str(value).ljust(15)} |")
        print(horizontal_line)

    slow_actions = SlowActions()
    fast_actions = FastActions()

    # Cria e checa o ambiente
    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=False,
        actions_cfg=fast_actions,
    )
    print("\nCheck environment ...")
    check_env(env)

    if num_envs == 1:
        obs, _ = env.reset()

        if robot_config.path_model != "None":
            model = PPO.load(robot_config.path_model)
        else:
            model = None

        rewards = []

        pbar = trange(max_steps, desc="Probe envs", unit="step")
        for step in pbar:
            if robot_config.path_model != "None" and model is not None:
                actions, _ = model.predict(obs)
            else:
                actions = env.action_space.sample()

            obs, reward, terminated, truncated, infos = env.step(actions)

            # armazena as recompensas
            print("Step: ", step, "Reward: ", reward)
            rewards.append(reward)

    else:
        # Cria o ambiente vetorizado
        env = make_vect_envs(
            num_envs=num_envs,
            robot_config=robot_config,
            sensor_config=sensor_config,
            env_config=env_config,
            render_config=render_config,
            use_render=False,
        )

        obs, info = env.reset()
        ep_rewards = np.zeros(num_envs)
        ep_lengths = np.zeros(num_envs)

        completed_rewards = []
        completed_lengths = []
        obstacles_scores = []
        collision_scores = []
        orientation_scores = []
        progress_scores = []
        time_scores = []
        total_rewards = []
        actions_list = []
        dists_list = []
        alphas_list = []
        min_lidars_list = []
        max_lidars_list = []

        if robot_config.path_model != "None":
            model = PPO.load(robot_config.path_model)

        pbar = trange(max_steps, desc="Probe envs", unit="step")

        for i in pbar:
            if robot_config.path_model != "None":
                actions, _states = model.predict(obs)
            else:
                actions = env.action_space.sample()
            obs, rewards, terminated, truncated, infos = env.step(actions)
            ep_rewards += np.array(rewards)
            ep_lengths += 1

            if infos is not None:
                for env_idx in range(num_envs):
                    obstacles_scores.append(infos["obstacle"][env_idx])
                    collision_scores.append(infos["collision_score"][env_idx])
                    orientation_scores.append(infos["orientation_score"][env_idx])
                    progress_scores.append(infos["progress_score"][env_idx])
                    time_scores.append(infos["time_score"][env_idx])
                    total_rewards.append(rewards[env_idx])
                    actions_list.append(infos["action"][env_idx])
                    dists_list.append(infos["dist"][env_idx])
                    alphas_list.append(infos["alpha"][env_idx])
                    min_lidars_list.append(infos["min_lidar"][env_idx])
                    max_lidars_list.append(infos["max_lidar"][env_idx])

            done = np.logical_or(terminated, truncated)
            done_indices = np.where(done)[0]

            if done_indices.size > 0:
                for idx in done_indices:
                    completed_rewards.append(ep_rewards[idx])
                    completed_lengths.append(ep_lengths[idx])

                    ep_rewards[idx] = 0
                    ep_lengths[idx] = 0

            if len(completed_rewards) > 0 and len(completed_lengths) > 0:
                avg_reward = np.mean(completed_rewards[-100:])
                avg_length = np.mean(completed_lengths[-100:])
            else:
                avg_reward = 0
                avg_length = 0

            pbar.set_postfix(
                {
                    "Ep Comp.": len(completed_rewards),
                    "Mean Reward(100)": f"{avg_reward:.2f}",
                    "Mean lenght(100)": f"{avg_length:.2f}",
                }
            )

        for idx in range(num_envs):
            if ep_lengths[idx] > 0:
                completed_rewards.append(ep_rewards[idx])
                completed_lengths.append(ep_lengths[idx])

        env.close()

        completed_rewards = np.array(completed_rewards)
        completed_lengths = np.array(completed_lengths)

        steps_range = list(range(1, len(total_rewards) + 1))
        step_metrics = [
            ("Obstacles Score", obstacles_scores, "brown"),
            ("Collision Score", collision_scores, "red"),
            ("Orientation Score", orientation_scores, "green"),
            ("Progress Score", progress_scores, "blue"),
            ("Time Score", time_scores, "orange"),
            ("Total Reward", total_rewards, "purple"),
            ("Action", actions_list, "blue"),
            ("Distance", dists_list, "cyan"),
            ("Alpha", alphas_list, "magenta"),
            ("Min Lidar", min_lidars_list, "yellow"),
            ("Max Lidar", max_lidars_list, "black"),
        ]

        total_subplots = len(step_metrics) + 1
        cols = 2
        rows = (total_subplots + cols - 1) // cols  # 6

        plt.figure(figsize=(10, 5 * rows))

        for idx, (title, data, color) in enumerate(step_metrics, start=1):
            ax = plt.subplot(rows, cols, idx)
            ax.plot(
                steps_range,
                data,
                label=title,
                color=color,
                linestyle="-",
                linewidth=1.5,
            )
            ax.set_ylabel(title, fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.tick_params(axis="x", labelsize=6)
            ax.tick_params(axis="y", labelsize=6)

            mean_val = np.mean(data)
            min_val = np.min(data)
            max_val = np.max(data)

            ax.text(
                0.5,
                -0.25,
                f"Média: {mean_val:.4f} | Mínimo: {min_val:.4f} | Máximo: {max_val:.4f}",
                transform=ax.transAxes,
                ha="center",
                fontsize=6,
            )

        ax_ep = plt.subplot(rows, cols, total_subplots)
        episodes_range = range(1, len(completed_rewards) + 1)

        ax_ep.plot(
            episodes_range, completed_rewards, label="Completed Rewards", color="black"
        )
        ax_ep.plot(
            episodes_range, completed_lengths, label="Completed Lengths", color="gray"
        )

        ax_ep.set_ylabel("Geral", fontsize=8)
        ax_ep.legend(fontsize=6)
        ax_ep.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax_ep.tick_params(axis="x", labelsize=6)
        ax_ep.tick_params(axis="y", labelsize=6)

        mean_rewards = np.mean(completed_rewards)
        min_rewards = np.min(completed_rewards)
        max_rewards = np.max(completed_rewards)

        mean_lengths = np.mean(completed_lengths)
        min_lengths = np.min(completed_lengths)
        max_lengths = np.max(completed_lengths)

        ax_ep.text(
            0.5,
            -0.4,
            f"Rewards -> Média: {mean_rewards:.4f} | Mínimo: {min_rewards:.4f} | Máximo: {max_rewards:.4f}\n"
            f"Lengths -> Média: {mean_lengths:.4f} | Mínimo: {min_lengths:.4f} | Máximo: {max_lengths:.4f}",
            transform=ax_ep.transAxes,
            ha="center",
            fontsize=6,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()


def probe_training(
    num_envs, max_steps, robot_config, sensor_config, env_config, render_config
):

    set_seed(13)

    # Exibe as configurações
    config_dict = {
        "Robot Config": robot_config.__dict__,
        "Sensor Config": sensor_config.__dict__,
        "Env Config": env_config.__dict__,
        "Render Config": render_config.__dict__,
    }
    table_width = 45
    horizontal_line = "-" * table_width
    print(horizontal_line)
    for name, cfg in config_dict.items():
        print(f"| {name + '/':<41} |")
        print(horizontal_line)
        for key, value in cfg.items():
            print(f"|    {key.ljust(20)} | {str(value).ljust(15)} |")
        print(horizontal_line)

    # Cria e checa o ambiente
    slow_actions = SlowActions()
    fast_actions = FastActions()

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=False,
        actions_cfg=fast_actions,
    )
    print("\nCheck environment ...")
    check_env(env)

    # NET_CONFIG = {
    #     "encoder_config": {'hidden_size': [32, 32]},  # Network head hidden size
    #     "head_config": {'hidden_size': [32]}      # Network head hidden size
    # }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "latent_dim": 16,
        "encoder_config": {"hidden_size": [40]},  # Observation encoder configuration
        "head_config": {"hidden_size": [40]},  # Network head configuration
    }

    INIT_HP = {
        "POP_SIZE": 2,  # Population size
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
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
    }

    env = make_vect_envs(
        num_envs=num_envs,
        robot_config=robot_config,
        sensor_config=sensor_config,
        env_config=env_config,
        render_config=render_config,
        use_render=False,
    )

    observation_space = env.single_observation_space
    action_space = env.single_action_space

    pop = create_population(
        algo="PPO",  # RL algorithm
        observation_space=observation_space,  # State dimension
        action_space=action_space,  # Action dimension
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,  # Number of vectorized envs
        device=device,
    )

    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.training.train_on_policy import train_on_policy

    MUTATION_PARAMS = {
        # Relative probabilities
        "NO_MUT": 0.4,  # No mutation
        "ARCH_MUT": 0.2,  # Architecture mutation
        "NEW_LAYER": 0.2,  # New layer mutation
        "PARAMS_MUT": 0.2,  # Network parameters mutation
        "ACT_MUT": 0,  # Activation layer mutation
        "RL_HP_MUT": 0.2,  # Learning HP mutation
        "MUT_SD": 0.1,  # Mutation strength
        "RAND_SEED": 1,  # Random seed
    }

    # tournament = TournamentSelection(
    #     tournament_size=INIT_HP['TOURN_SIZE'], # Tournament selection size
    #     elitism=INIT_HP['ELITISM'],            # Elitism in tournament selection
    #     population_size=INIT_HP['POP_SIZE'],   # Population size
    #     eval_loop=INIT_HP['EVAL_LOOP'],        # Evaluate using last N fitness scores
    # )

    # mutations = Mutations(
    #     no_mutation=MUTATION_PARAMS['NO_MUT'],                # No mutation
    #     architecture=MUTATION_PARAMS['ARCH_MUT'],             # Architecture mutation
    #     new_layer_prob=MUTATION_PARAMS['NEW_LAYER'],          # New layer mutation
    #     parameters=MUTATION_PARAMS['PARAMS_MUT'],             # Network parameters mutation
    #     activation=MUTATION_PARAMS['ACT_MUT'],                # Activation layer mutation
    #     rl_hp=MUTATION_PARAMS['RL_HP_MUT'],                   # Learning HP mutation
    #     mutation_sd=MUTATION_PARAMS['MUT_SD'],                # Mutation strength
    #     rand_seed=MUTATION_PARAMS['RAND_SEED'],               # Random seed
    #     device=device,
    # )

    trained_pop, pop_fitnesses = train_on_policy(
        algo="PPO",
        env=env,  # Gym-style environment
        env_name="NaviEnv",  # Environment name
        pop=pop,  # Population of agents
        swap_channels=INIT_HP["CHANNELS_LAST"],  # Swap image channel from last to first
        max_steps=1000000,  # Max number of training steps
        evo_steps=10000,  # Evolution frequency
        eval_steps=None,  # Number of steps in evaluation episode
        eval_loop=1,  # Number of evaluation episodes
        target=None,  # Target score for early stopping
        wb=True,  # Weights and Biases tracking
        verbose=True,
        wandb_api_key="fb372890f5180a16a9cd2df5b9558e55493cd16c",
        # tournament=tournament,                     # Tournament selection object
        # mutation=mutations,
        checkpoint=50000,
    )
