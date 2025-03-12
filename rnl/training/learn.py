import matplotlib.pyplot as plt
import numpy as np
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tqdm import trange

from rnl.agents.evaluator import LLMTrainingEvaluator
from rnl.configs.actions import get_actions_class
from rnl.configs.config import (
    EnvConfig,
    NetworkConfig,
    ProbeEnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.configs.rewards import RewardConfig
from rnl.engine.utils import clean_info, set_seed
from rnl.engine.vector import make_vect_envs
from rnl.environment.env import NaviEnv
from rnl.training.train import training_loop


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

    actions_type = get_actions_class("BalancedActions")()
    reward_instance = RewardConfig(
        reward_type="all",
        params={
            "scale_orientation": 0.003,
            "scale_distance": 0.1,
            "scale_time": 0.01,
            "scale_obstacle": 0.001,
        },
        description="Reward baseado em todos os fatores",
    )
    mode = "hard"

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=True,
        actions_cfg=actions_type,
        reward_cfg=reward_instance,
        mode=mode,
    )

    env.render()


def probe_envs(
    num_envs, max_steps, robot_config, sensor_config, env_config, render_config, seed
):
    # set_seed(seed)

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

    actions_type = get_actions_class("BalancedActions")()
    reward_instance = RewardConfig(
        reward_type="all",
        params={
            "scale_orientation": 0.003,
            "scale_distance": 0.1,
            "scale_time": 0.01,
            "scale_obstacle": 0.001,
        },
        description="Reward baseado em todos os fatores",
    )

    mode = "easy-01"

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=False,
        actions_cfg=actions_type,
        reward_cfg=reward_instance,
        mode=mode,
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
        env = make_vect_envs(
            num_envs=num_envs,
            robot_config=robot_config,
            sensor_config=sensor_config,
            env_config=env_config,
            render_config=render_config,
            use_render=False,
            actions_type=actions_type,
            reward_type=reward_instance,
            mode=mode
        )

        obs, info = env.reset()
        ep_rewards = np.zeros(num_envs)
        ep_lengths = np.zeros(num_envs)

        completed_rewards = []
        completed_lengths = []
        obstacles_scores = []
        orientation_scores = []
        progress_scores = []
        time_scores = []
        total_rewards = []
        actions_list = []
        dists_list = []
        alphas_list = []
        min_lidars_list = []
        max_lidars_list = []

        model = None
        if robot_config.path_model != "None":
            model = PPO.load(robot_config.path_model)

        pbar = trange(max_steps, desc="Probe envs", unit="step")

        for i in pbar:
            if robot_config.path_model != "None" and model is not None:
                actions, _states = model.predict(obs)
            else:
                actions = env.action_space.sample()
            obs, rewards, terminated, truncated, infos = env.step(actions)
            ep_rewards += np.array(rewards)
            ep_lengths += 1

            infos = clean_info(infos)

            if infos is not None and infos:
                for env_idx in range(num_envs):
                    obstacles_scores.append(infos["obstacle_score"][env_idx])
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


def training(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    trainer_config: TrainerConfig,
    network_config: NetworkConfig,
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
    for name, cfg in config_dict.items():
        print(f"| {name + '/':<41} |")
        print(horizontal_line)
        for key, value in cfg.items():
            print(f"|    {key.ljust(20)} | {str(value).ljust(15)} |")
        print(horizontal_line)

    actions_type = get_actions_class("BalancedActions")()
    reward_instance = RewardConfig(
        reward_type="time",
        params={
            "scale_orientation": 0.003,
            "scale_distance": 0.1,
            "scale_time": 0.01,
            "scale_obstacle": 0.001,
        },
        description="Reward baseado em todos os fatores",
    )
    mode = "easy-01"

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=False,
        actions_cfg=actions_type,
        reward_cfg=reward_instance,
        mode=mode,
    )
    print("\nCheck environment ...")
    check_env(env)

    NET_CONFIG = {
        "latent_dim": 16,
        "encoder_config": {
            "hidden_size": [network_config.hidden_size[0]]
        },  # Observation encoder configuration
        "head_config": {
            "hidden_size": [network_config.hidden_size[1]]
        },  # Network head configuration
    }

    INIT_HP = {
        "POP_SIZE": 4,  # Population size
        "BATCH_SIZE": trainer_config.batch_size,  # Batch size
        "LR": trainer_config.lr,  # Learning rate
        "LEARN_STEP": trainer_config.learn_step,  # Learning frequency
        "GAMMA": 0.99,  # Discount factor
        "GAE_LAMBDA": trainer_config.gae_lambda,  # Lambda for general advantage estimation
        "ACTION_STD_INIT": trainer_config.action_std_init,  # Initial action standard deviation
        "CLIP_COEF": trainer_config.clip_coef,  # Surrogate clipping coefficient
        "ENT_COEF": trainer_config.ent_coef,  # Entropy coefficient
        "VF_COEF": trainer_config.vf_coef,  # Value function coefficient
        "MAX_GRAD_NORM": trainer_config.max_grad_norm,  # Maximum norm for gradient clipping
        "TARGET_KL": None,  # Target KL divergence threshold
        "UPDATE_EPOCHS": trainer_config.update_epochs,  # Number of policy update epochs
        "TOURN_SIZE": 2,
        "ELITISM": True,  # Elitism in tournament selection
        "CHANNELS_LAST": False,
    }

    MUTATION_PARAMS = {
        "NO_MUT": 0.4,  # No mutation
        "ARCH_MUT": 0.2,  # Architecture mutation
        "NEW_LAYER": 0.2,  # New layer mutation
        "PARAMS_MUT": 0.2,  # Network parameters mutation
        "ACT_MUT": 0,  # Activation layer mutation
        "RL_HP_MUT": 0.2,  # Learning HP mutation
        "MUT_SD": 0.1,  # Mutation strength
        "RAND_SEED": trainer_config.seed,  # Random seed
    }

    env = make_vect_envs(
        num_envs=trainer_config.num_envs,
        robot_config=robot_config,
        sensor_config=sensor_config,
        env_config=env_config,
        render_config=render_config,
        use_render=False,
        actions_type=actions_type,
        reward_type=reward_instance,
        mode=mode,
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
        num_envs=trainer_config.num_envs,  # Number of vectorized envs
        device=trainer_config.device,
    )

    tournament = TournamentSelection(
        tournament_size=INIT_HP["TOURN_SIZE"],  # Tournament selection size
        elitism=INIT_HP["ELITISM"],  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        no_mutation=MUTATION_PARAMS["NO_MUT"],  # No mutation
        architecture=MUTATION_PARAMS["ARCH_MUT"],  # Architecture mutation
        new_layer_prob=MUTATION_PARAMS["NEW_LAYER"],  # New layer mutation
        parameters=MUTATION_PARAMS["PARAMS_MUT"],  # Network parameters mutation
        activation=MUTATION_PARAMS["ACT_MUT"],  # Activation layer mutation
        rl_hp=MUTATION_PARAMS["RL_HP_MUT"],  # Learning HP mutation
        mutation_sd=MUTATION_PARAMS["MUT_SD"],  # Mutation strength
        rand_seed=MUTATION_PARAMS["RAND_SEED"],  # Random seed
        device=trainer_config.device,
    )

    evaluator = LLMTrainingEvaluator(evaluator_api_key=trainer_config.llm_api_key)

    env = training_loop(
        trainer_config.use_agents,
        trainer_config.num_envs,
        trainer_config.max_timestep_global,
        trainer_config.evo_steps,
        None,
        1,
        env,
        pop,
        tournament,
        mutations,
        evaluator,
        robot_config,
        sensor_config,
        env_config,
        render_config,
        actions_type,
        reward_instance,
        trainer_config.use_wandb,
        trainer_config.save_path,
        trainer_config.elite_path,
        trainer_config.checkpoint,
        trainer_config.overwrite_checkpoints,
        INIT_HP,
        MUTATION_PARAMS,
        trainer_config.wandb_api_key,
        trainer_config.save_elite,
    )
