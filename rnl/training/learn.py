import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_checker import check_env
from tqdm import trange
from rnl.agents.evaluator import LLMTrainingEvaluator
from rnl.configs.config import (
    EnvConfig,
    NetworkConfig,
    ProbeEnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.configs.strategys import get_strategy_dict
from rnl.training.callback import DynamicTrainingCallback
from rnl.engine.utils import clean_info
from rnl.engine.vector import make_vect_envs
from rnl.environment.env import NaviEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from wandb.integration.sb3 import WandbCallback
from rnl.configs.rewards import RewardConfig
import wandb

ENV_TYPE = "train-mode"
PORCENTAGE_OBSTACLE = 40.0
MAP_SIZE = 2.0
POLICY = "PPO"
REWARD_TYPE = RewardConfig(
    reward_type="time",
    params={
        "scale_orientation": 0.02,
        "scale_distance": 0.06,
        "scale_time": 0.01,
        "scale_obstacle": 0.004,
    },
    description="Reward baseado em todos os fatores",
)

def training(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    trainer_config: TrainerConfig,
    network_config: NetworkConfig,
):
    extra_info = {
        "Type mode": ENV_TYPE,
        "Type policy": POLICY,
        "reward_type": REWARD_TYPE.reward_type,
        "scale_orientation": REWARD_TYPE.params["scale_orientation"],
        "scale_distance": REWARD_TYPE.params["scale_distance"],
        "scale_time": REWARD_TYPE.params["scale_time"],
        "scale_obstacle": REWARD_TYPE.params["scale_obstacle"],
    }

    config_dict = {
        "Trainer Config": trainer_config.__dict__,
        "Network Config": network_config.__dict__,
        "Robot Config": robot_config.__dict__,
        "Sensor Config": sensor_config.__dict__,
        "Env Config": env_config.__dict__,
        "Render Config": render_config.__dict__,
    }

    config_dict.update(extra_info)

    table_width = 45
    horizontal_line = "-" * table_width
    print(horizontal_line)

    for config_name, config_values in config_dict.items():
        print(f"| {config_name + '/':<41} |")
        print(horizontal_line)
        if isinstance(config_values, dict):
            for key, value in config_values.items():
                print(f"|    {key.ljust(20)} | {str(value).ljust(15)} |")
        else:
            print(f"|    {str(config_values).ljust(20)} | {' '.ljust(15)}|")
        print(horizontal_line)

    if trainer_config.use_wandb:
        run = wandb.init(
            name="mode_sb3",
            project=trainer_config.name,
            config=config_dict,
            sync_tensorboard=False,
            monitor_gym=True,
            save_code=False,
        )

    evaluator = LLMTrainingEvaluator(evaluator_api_key=trainer_config.llm_api_key)

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=False,
        mode=ENV_TYPE,
        type_reward=REWARD_TYPE,
        porcentage_obstacle=PORCENTAGE_OBSTACLE,
        map_size=MAP_SIZE,
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

    policy_kwargs_off_policy = dict(
            activation_fn=activation_fn,
            net_arch=[network_config.hidden_size[0], network_config.hidden_size[1]],
        )

    def make_env():
        env = NaviEnv(
            robot_config,
            sensor_config,
            env_config,
            render_config,
            use_render=False,
            mode=ENV_TYPE,
            type_reward=REWARD_TYPE,
            porcentage_obstacle=PORCENTAGE_OBSTACLE,
            map_size=MAP_SIZE,
        )
        env = Monitor(env)
        return env

    # Parallel environments
    vec_env = make_vec_env(make_env, n_envs=trainer_config.num_envs)

    print("\nInitiate PPO training ...")

    if trainer_config.use_wandb:
        model = None
        if POLICY == "PPO":
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
        elif POLICY == "A2C":
                model = A2C(
                    "MlpPolicy",
                    vec_env,
                    verbose=1,
                    learning_rate=trainer_config.lr,
                    n_steps=trainer_config.learn_step,
                    gae_lambda=trainer_config.gae_lambda,
                    ent_coef=trainer_config.ent_coef,
                    vf_coef=trainer_config.vf_coef,
                    max_grad_norm=trainer_config.max_grad_norm,
                    seed=trainer_config.seed,
                    policy_kwargs=policy_kwargs_on_policy,
                    device=trainer_config.device,
                )
                print("\nInitiate A2C training ...")

        elif POLICY == "DQN":
            env = DummyVecEnv([make_env])
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=trainer_config.lr,
                batch_size=trainer_config.batch_size,
                buffer_size=1000000,
                verbose=1,
                device=trainer_config.device,
                policy_kwargs=policy_kwargs_off_policy,
                seed=trainer_config.seed,
            )

            print("\nInitiate DQN training ...")

        if model is not None:
            model.learn(
                total_timesteps=trainer_config.max_timestep_global,
                callback=WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=trainer_config.checkpoint_path,
                    verbose=2,
                ),
            )
            run.finish()

    else:
        if POLICY == "PPO":
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

        elif POLICY == "A2C":
                model = A2C(
                    "MlpPolicy",
                    vec_env,
                    verbose=1,
                    learning_rate=trainer_config.lr,
                    n_steps=trainer_config.learn_step,
                    gae_lambda=trainer_config.gae_lambda,
                    ent_coef=trainer_config.ent_coef,
                    vf_coef=trainer_config.vf_coef,
                    max_grad_norm=trainer_config.max_grad_norm,
                    seed=trainer_config.seed,
                    policy_kwargs=policy_kwargs_on_policy,
                    device=trainer_config.device,
                )
                print("\nInitiate A2C training ...")

        elif POLICY == "DQN":
            env = DummyVecEnv([make_env])
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=trainer_config.lr,
                batch_size=trainer_config.batch_size,
                buffer_size=1000000,
                verbose=1,
                device=trainer_config.device,
                policy_kwargs=policy_kwargs_off_policy,
                seed=trainer_config.seed,
            )

            print("\nInitiate DQN training ...")

        if trainer_config.use_agents:
            callback = DynamicTrainingCallback(
                evaluator=evaluator,
                justificativas_history=[],
                get_strategy_dict_func=get_strategy_dict,
                get_parameter_train=config_dict,
                check_freq=100
            )

        else:
            callback = None

        model.learn(total_timesteps=trainer_config.max_timestep_global, callback=callback)

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

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=True,
        mode=ENV_TYPE,
        type_reward=REWARD_TYPE,
        porcentage_obstacle=PORCENTAGE_OBSTACLE,
        map_size=MAP_SIZE,
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

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=False,
        mode=ENV_TYPE,
        type_reward=REWARD_TYPE,
        porcentage_obstacle=PORCENTAGE_OBSTACLE,
        map_size=MAP_SIZE,
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

        for step in range(max_steps):
            if robot_config.path_model != "None" and model is not None:
                actions, _ = model.predict(obs)
            else:
                actions = env.action_space.sample()

            obs, reward, terminated, truncated, infos = env.step(actions)
            print("Step:", step, "reward: ", reward)
            rewards.append(reward)

            if terminated or truncated:
                obs, _ = env.reset()

    else:
        env = make_vect_envs(
            num_envs=num_envs,
            robot_config=robot_config,
            sensor_config=sensor_config,
            env_config=env_config,
            render_config=render_config,
            use_render=False,
            mode=ENV_TYPE,
            type_reward=REWARD_TYPE,
            porcentage_obstacle=PORCENTAGE_OBSTACLE,
            map_size=MAP_SIZE,
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
