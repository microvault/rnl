import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.dqn.policies import DQNPolicy
from rnl.engine.vector import make_vect_envs
import matplotlib as plt
import csv
import wandb
from rnl.configs.config import (
    EnvConfig,
    NetworkConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.environment.env import NaviEnv


def training(
    trainer_config: TrainerConfig,
    network_config: NetworkConfig,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    pretrained_model: bool,
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
        actions = env.action_space.sample()

        next_state, rewards, terminated, truncated, info = env.step(actions)
        steps += 1

        ep_rewards += np.array(rewards)
        ep_lengths += 1  # Incrementa o comprimento do episódio

        done = np.logical_or(terminated, truncated)
        done_indices = np.where(done)[0]

        if done_indices.size > 0:
            for idx in done_indices:
                completed_rewards.append(ep_rewards[idx])
                completed_lengths.append(ep_lengths[idx])

                ep_rewards[idx] = 0
                ep_lengths[idx] = 0

        if len(completed_rewards) > 0 and len(completed_lengths) > 0:
            avg_reward = np.mean(
                completed_rewards[-100:]
            )
            avg_length = np.mean(
                completed_lengths[-100:]
            )
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

    for idx in range(num_envs):
        if ep_lengths[idx] > 0:
            completed_rewards.append(ep_rewards[idx])
            completed_lengths.append(ep_lengths[idx])

    env.close()

    completed_rewards = np.array(completed_rewards)
    completed_lengths = np.array(completed_lengths)

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

    steps = list(range(1, len(rewards) + 1))

    components = [
        ("Obstacles Score", obstacles_scores, "brown"),
        ("Collision Score", collision_scores, "red"),
        ("Orientation Score", orientation_scores, "green"),
        ("Progress Score", progress_scores, "blue"),
        ("Time Score", time_scores, "orange"),
        ("Total Reward", rewards, "purple"),
        ("ep_len_mean", completed_lengths, "gray"),
        ("ep_rew_mean", completed_rewards, "black"),
    ]

    num_plots = len(components)
    cols = 2
    rows = (num_plots + cols - 1) // cols

    plt.figure(figsize=(10, 5 * rows))

    for idx, (title, data, color) in enumerate(components, 1):
        ax = plt.subplot(rows, cols, idx)
        ax.plot(steps, data, label=title, color=color, linestyle="-", linewidth=1.5)
        ax.set_ylabel(title, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        mean_val = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)

        ax.text(
            0.5,
            -0.25,
            f"Média: {mean_val:.2f} | Mínimo: {min_val:.2f} | Máximo: {max_val:.2f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
        )

    plt.tight_layout()

    plt.subplots_adjust(bottom=0.15)

    plt.show()
