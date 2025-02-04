import matplotlib.pyplot as plt
import numpy as np
import wandb
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from tqdm import trange
from wandb.integration.sb3 import WandbCallback

from rnl.configs.config import (
    EnvConfig,
    NetworkConfig,
    ProbeEnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
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

    env = NaviEnv(
        robot_config, sensor_config, env_config, render_config, use_render=False
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
        net_arch=dict(
            pi=[network_config.hidden_size[0], network_config.hidden_size[1]],
            qf=[network_config.hidden_size[0], network_config.hidden_size[1]],
        ),
    )

    def make_env():
        env = NaviEnv(
            robot_config, sensor_config, env_config, render_config, use_render=False
        )
        env = Monitor(env)
        return env

    print(trainer_config)

    # Parallel environments
    vec_env = make_vec_env(make_env, n_envs=trainer_config.num_envs)
    if trainer_config.algorithm == "PPO":
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

        print("\nInitiate PPO training ...")

    elif trainer_config.algorithm == "A2C":
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
            tensorboard_log=f"runs/{run.id}",
            device=trainer_config.device,
        )
        print("\nInitiate A2C training ...")

    elif trainer_config.algorithm == "DQN":
        env = DummyVecEnv([make_env])
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=trainer_config.lr,
            batch_size=trainer_config.batch_size,
            buffer_size=trainer_config.buffer_size,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device=trainer_config.device,
            policy_kwargs=policy_kwargs_off_policy,
            seed=trainer_config.seed,
        )

        print("\nInitiate DQN training ...")

    else:
        print("Invalid algorithm")

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

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=True,
    )
    obs_space = env.observation_space
    state_dim = obs_space.shape
    print("States dim: ", state_dim)

    action_space = env.action_space
    action_dim = action_space.n
    print("Action dim:", action_dim)

    env.reset()
    env.render()


def probe_envs(
    num_envs: int,
    max_steps: int,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
):

    assert num_envs >= 1, "num_envs must be greater than 1"

    probe_config = ProbeEnvConfig(
        num_envs=num_envs,
        max_steps=max_steps,
    )

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

    for config_name, config_values in config_dict.items():
        print(f"| {config_name + '/':<41} |")
        print(horizontal_line)
        for key, value in config_values.items():
            print(f"|    {key.ljust(20)} | {str(value).ljust(15)} |")
        print(horizontal_line)

    env = NaviEnv(
        robot_config, sensor_config, env_config, render_config, use_render=False
    )
    print("\nCheck environment ...")
    check_env(env)

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
    states_list = []

    if robot_config.path_model != "None":
        model = PPO.load(robot_config.path_model)

    pbar = trange(max_steps, desc="Probe envs", unit="step")

    for i in pbar:
        if robot_config.path_model != "None":
            actions, _states = model.predict(obs)
        else:
            actions = env.action_space.sample()
        obs, rewards, terminated, truncated, infos = env.step(actions)
        # if 'final_observation' in infos or '_final_observation' in infos or 'final_info' in infos or '_final_info' in infos:
        #     infos = list(infos)
        #     for i in range(num_envs):
        #         for k in ["final_observation", "_final_observation", "final_info", "_final_info"]:
        #             if k in infos[i]:
        #                 del infos[i][k]

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
                    states_list.append(infos["states"][env_idx])

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
            steps_range, data, label=title, color=color, linestyle="-", linewidth=1.5
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
