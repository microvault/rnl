import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from torch import nn
from tqdm import trange
from wandb.integration.sb3 import WandbCallback
import wandb
from rnl.algorithms.ppo import PPO as RL_PPO
from rnl.configs.config import (
    EnvConfig,
    NetworkConfig,
    ProbeEnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
import torch
from torch.distributions import Categorical
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

    def make_env():
        env = NaviEnv(
            robot_config, sensor_config, env_config, render_config, use_render=False
        )
        env = Monitor(env)
        return env

    # Parallel environments
    vec_env = make_vec_env(make_env, n_envs=trainer_config.num_envs)
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

    if trainer_config.use_wandb:
        model.learn(
            total_timesteps=trainer_config.max_timestep_global,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"model_{trainer_config.checkpoint}_{trainer_config.algorithm}/{run.id}",
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

    # Crie seu ambiente único; substitua NaviEnv pela sua classe
    # env = NaviEnv(robot_config, sensor_config, env_config, render_config, use_render=False)
    # Obter dimensão do estado e número de ações
    # Se env.reset() retornar tuple, extraia a primeira parte
    # obs = env.reset()
    # if isinstance(obs, tuple):
    #     obs = obs[0]
    # input_dim = env.observation_space.shape[0]
    # output_dim = env.action_space.n
    # hidden_dims = [64, 64]

    # model = RL_PPO(input_dim, hidden_dims, output_dim)

    # episode_reward = 0
    # episode_rewards_history = []
    # num_updates = 200000
    # T_horizon     = 20

    # # Configurar device
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps")

    # print(f"Usando device: {device}")
    # BATCH_SIZE = 1024  # Definindo o tamanho do batch

    # update_count = 0
    # for update in range(num_updates):
    #     for t in range(T_horizon):
    #         # Se obs for um dict, extraia a chave "state".
    #         # Se for um tuple, use a primeira parte.
    #         if isinstance(obs, dict):
    #             try:
    #                 state = np.array(obs["state"])
    #             except KeyError:
    #                 raise KeyError("A chave 'state' não foi encontrada na observação.")
    #         elif isinstance(obs, tuple):
    #             state = np.array(obs[0])
    #         else:
    #             state = np.array(obs)

    #         # Cria um tensor com dimensão de batch (1, input_dim)
    #         state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
    #         action_probs = model.pi(state_tensor, softmax_dim=1)
    #         dist = Categorical(action_probs)
    #         action = dist.sample().item()
    #         prob = action_probs[0, action].item()

    #         next_obs, reward, done, info, _ = env.step(action)
    #         # Se next_obs for tuple, extraia a observação
    #         if isinstance(next_obs, tuple):
    #             next_obs = next_obs[0]
    #         if isinstance(next_obs, dict):
    #             next_state = np.array(next_obs["state"])
    #         else:
    #             next_state = np.array(next_obs)

    #         model.put_data((state, action, reward, next_state, prob, done))
    #         episode_reward += reward
    #         obs = next_obs

    #         if done:
    #             episode_rewards_history.append(episode_reward)
    #             episode_reward = 0
    #             obs = env.reset()
    #             if isinstance(obs, tuple):
    #                 obs = obs[0]

    #     # Atualiza o modelo após T_horizon passos
    #     if len(model.data) >= BATCH_SIZE:
    #         print("Train model")
    #         model.train_net()
    #         update_count += 1

    #     # Imprime a média de recompensa a cada 100 episódios
    #     if len(episode_rewards_history) >= 10 and len(episode_rewards_history) % 10 == 0:
    #         avg_reward = np.mean(episode_rewards_history[-10:])
    #         print(f"Update {update} - Últimos {len(episode_rewards_history)} episódios: Média de recompensa: {avg_reward:.2f}")

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
    steps_to_goal_list = []
    steps_below_thresh_list = []
    steps_to_collision_list = []
    turn_left_list = []
    turn_right_list = []

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

        metrics = {
            "obstacle": obstacles_scores,
            "collision_score": collision_scores,
            "orientation_score": orientation_scores,
            "progress_score": progress_scores,
            "time_score": time_scores,
            "action": actions_list,
            "dist": dists_list,
            "alpha": alphas_list,
            "min_lidar": min_lidars_list,
            "max_lidar": max_lidars_list,
            "states": states_list,
            "steps_to_goal": steps_to_goal_list,
            "steps_below_threshold": steps_below_thresh_list,
            "steps_to_collision": steps_to_collision_list,
            "turn_left_count": turn_left_list,
            "turn_right_count": turn_right_list,
        }

        if infos is not None:
            for env_idx in range(num_envs):
                for key, lst in metrics.items():
                    # Se a chave não existir, usa uma lista default com 0.0
                    value_list = infos.get(key, [0.0] * num_envs)
                    lst.append(value_list[env_idx])

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

    if render_config.debug:

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
            ("Steps to Goal", steps_to_goal_list, "purple"),
            ("Steps Below Threshold", steps_below_thresh_list, "magenta"),
            ("Steps to Collision", steps_to_collision_list, "orange"),
            ("Turns Left", turn_left_list, "cyan"),
            ("Turns Right", turn_right_list, "lime"),
        ]

        total_subplots = len(step_metrics) + 1
        cols = 2
        rows = (total_subplots + cols - 1) // cols  # 6

        plt.figure(figsize=(10, 5 * rows))

        for idx, (title, data, color) in enumerate(step_metrics, start=1):
            ax = plt.subplot(rows, cols, idx)
            # Cria um eixo x baseado no tamanho da lista "data"
            steps_range = list(range(1, len(data) + 1))
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
            ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(axis="y", labelsize=6)

            mean_val, min_val, max_val = safe_stats(data)

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
        ax_ep.tick_params(axis="x", labelbottom=False)
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


def safe_stats(data):
    clean_data = [v for v in data if v is not None]
    if not clean_data:
        return 0.0, 0.0, 0.0
    return np.mean(clean_data), np.min(clean_data), np.max(clean_data)
