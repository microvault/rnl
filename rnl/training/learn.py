import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import TRPO, RecurrentPPO, QRDQN
from stable_baselines3 import A2C, DQN, PPO

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from tqdm import trange
import os
import random
import wandb
from rnl.network.model import CustomActorCriticPolicy
from rnl.agents.evaluate import evaluate_agent, statistics

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
from rnl.engine.utils import clean_info, print_config_table, set_seed
from rnl.engine.vector import make_vect_envs, make_vect_envs_norm
from rnl.environment.env import NaviEnv
from rnl.training.callback import DynamicTrainingCallback

TYPE = "turn"
OBSTACLE_PERCENTAGE = 20.0
MAP_SIZE = 5.0
POLICY = "PPO"
REWARD_TYPE = RewardConfig(
    params={
        "scale_orientation": 0.0,  # 0.02
        "scale_distance": 0.0,  # 0.06
        "scale_time": 0.01,  # 0.01
        "scale_obstacle": 0.0,  # 0.004
    },
)


def training(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    trainer_config: TrainerConfig,
    network_config: NetworkConfig,
    reward_config: RewardConfig,
    print_parameter: bool,
    train: bool,
):

    extra_info = {
        "Type mode": env_config.type,
        "Type policy": trainer_config.policy_type,
        "scale_orientation": reward_config.params["scale_orientation"],
        "scale_distance": reward_config.params["scale_distance"],
        "scale_time": reward_config.params["scale_time"],
        "scale_obstacle": reward_config.params["scale_obstacle"],
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

    if print_parameter:
        print_config_table(config_dict)

    run = None
    if trainer_config.use_wandb:
        if trainer_config.wandb_mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(
            name="rnl-test",
            project=trainer_config.name,
            config=config_dict,
            mode=trainer_config.wandb_mode,
            sync_tensorboard=False,
            monitor_gym=True,
            save_code=True,
        )

    policy_kwargs_on_policy = None
    policy_kwargs_on_policy_recurrent = None
    policy_kwargs_off_policy = None

    if network_config.type_model == CustomActorCriticPolicy:
        policy_kwargs_on_policy = dict(
            last_layer_dim_pi=network_config.hidden_size[0],
            last_layer_dim_vf=network_config.hidden_size[1],
        )
    else:
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

        policy_kwargs_on_policy_recurrent = dict(
            activation_fn=activation_fn,
            net_arch=dict(
                pi=[network_config.hidden_size[0], network_config.hidden_size[1]],
                vf=[network_config.hidden_size[0], network_config.hidden_size[1]],
            ),
            n_lstm_layers=1,
            lstm_hidden_size=32,
        )

        policy_kwargs_off_policy = dict(
            activation_fn=activation_fn,
            net_arch=[network_config.hidden_size[0], network_config.hidden_size[1]],
        )

    verbose_value = 0 if not trainer_config.verbose else 1
    model = None

    if train:
        def make_env():
            env = NaviEnv(
                robot_config,
                sensor_config,
                env_config,
                render_config,
                use_render=False,
                mode=env_config.type,
                type_reward=reward_config,
            )
            env = Monitor(env)
            return env

        vec_env = make_vec_env(make_env, n_envs=trainer_config.num_envs)
    else:
        vec_env = make_vect_envs(
            num_envs=trainer_config.num_envs,
            robot_config=robot_config,
            sensor_config=sensor_config,
            env_config=env_config,
            render_config=render_config,
            use_render=False,
            mode=env_config.type,
            type_reward=reward_config,
        )

        task_pool = ("long", "turn", "avoid")

        env_config.type = (
            np.random.choice(task_pool) if env_config.type == "random" else env_config.type
        )

        def make_env():
            env = NaviEnv(
                robot_config,
                sensor_config,
                env_config,
                render_config,
                use_render=False,
                mode=env_config.type,
                type_reward=reward_config,
            )
            env = Monitor(env)
            return env

    if trainer_config.pretrained != "None":
        model = PPO.load(trainer_config.pretrained)
    else:
        if policy_kwargs_on_policy is not None or policy_kwargs_on_policy_recurrent is not None or policy_kwargs_off_policy is not None:
            if trainer_config.policy_type == "TRPO":
                model = TRPO(
                    policy=network_config.type_model,
                    env=vec_env,
                    batch_size=trainer_config.batch_size,
                    verbose=verbose_value,
                    learning_rate=trainer_config.lr,
                    policy_kwargs=policy_kwargs_on_policy,
                    n_steps=trainer_config.learn_step,
                    device=trainer_config.device,
                    seed=trainer_config.seed,
                )
            if trainer_config.policy_type == "RecurrentPPO":
                print("RecurrentPPO")
                model = RecurrentPPO(
                    "MlpLstmPolicy",
                    vec_env,
                    batch_size=trainer_config.batch_size,
                    verbose=verbose_value,
                    learning_rate=trainer_config.lr,
                    policy_kwargs=policy_kwargs_on_policy_recurrent,
                    n_steps=trainer_config.learn_step,
                    vf_coef=trainer_config.vf_coef,
                    ent_coef=trainer_config.ent_coef,
                    device=trainer_config.device,
                    max_grad_norm=trainer_config.max_grad_norm,
                    n_epochs=trainer_config.update_epochs,
                    seed=trainer_config.seed,
                )
            elif trainer_config.policy_type == "PPO":
                model = PPO(
                    policy=network_config.type_model,
                    env=vec_env,
                    batch_size=trainer_config.batch_size,
                    verbose=verbose_value,
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
            elif trainer_config.policy_type == "A2C":
                model = A2C(
                    policy=network_config.type_model,
                    env=vec_env,
                    verbose=verbose_value,
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
            elif trainer_config.policy_type == "DQN":
                model = DQN(
                    'MlpPolicy',
                    DummyVecEnv([make_env]),
                    learning_rate=trainer_config.lr,
                    batch_size=trainer_config.batch_size,
                    buffer_size=1000000,
                    verbose=verbose_value,
                    device=trainer_config.device,
                    policy_kwargs=policy_kwargs_off_policy,
                    seed=trainer_config.seed,
                )
            elif trainer_config.policy_type == "QRDQN":
                model = QRDQN(
                    'MlpPolicy',
                    DummyVecEnv([make_env]),
                    learning_rate=trainer_config.lr,
                    batch_size=trainer_config.batch_size,
                    buffer_size=1000000,
                    verbose=verbose_value,
                    device=trainer_config.device,
                    policy_kwargs=policy_kwargs_off_policy,
                    seed=trainer_config.seed,
                )

    id = random.randint(0, 1000000)
    callback = DynamicTrainingCallback(
        check_freq=100,
        sample_checkpoint_freq=1000,
        run_id=str(id),
        wandb_run=run,
        save_checkpoint=trainer_config.checkpoint,
        model_save_path="checkpoints/",
        robot_config=robot_config,
        sensor_config=sensor_config,
        env_config=env_config,
        render_config=render_config,
        mode=env_config.type,
        type_reward=reward_config,
    )

    if model is not None:
        model.learn(
            total_timesteps=trainer_config.max_timestep_global,
            callback=callback,
        )

    if trainer_config.use_wandb:
        if run is not None:
            run.finish()

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=False,
        mode=env_config.type,
        type_reward=reward_config,
    )

    final_eval = evaluate_agent(model, env)

    metrics = {}
    if model is not None:
        infos_list = []
        for i in range(model.n_envs):
            env_info = model.get_env().env_method("get_infos", indices=i)[0]
            if env_info:
                infos_list.extend(env_info)

        stats = {}
        for campo in [
            "obstacle_score",
            "orientation_score",
            "progress_score",
            "time_score",
            "min_lidar",
        ]:
            if any(campo in info for info in infos_list):
                media, _, _, desvio = statistics(infos_list, campo)
                stats[campo + "_mean"] = media
                stats[campo + "_std"] = desvio

        metrics = stats

    eval_keys = [
        "success_percentage",
        "total_timesteps",
        "percentage_unsafe",
        "percentage_angular",
        "ep_mean_length",
        "avg_collision_steps",
        "avg_goal_steps",
    ]

    final_eval_dict = dict(zip(eval_keys, final_eval))

    merged_dict = {**metrics, **final_eval_dict}

    return merged_dict


def inference(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    reward_config: RewardConfig,
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
        "Type mode": type,
        "Reward Config": reward_config,
        "Robot Config": robot_config.__dict__,
        "Sensor Config": sensor_config.__dict__,
        "Env Config": env_config.__dict__,
        "Render Config": render_config.__dict__,
    }

    print_config_table(config_dict)

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=True,
        mode=env_config.type,
        type_reward=reward_config,
    )

    env.render()


def probe_envs(
    num_envs,
    max_steps,
    robot_config,
    sensor_config,
    env_config,
    render_config,
    seed,
    reward_config: RewardConfig,
    image=False,
):
    set_seed(seed)

    probe_config = ProbeEnvConfig(num_envs=num_envs, max_steps=max_steps)
    config_dict = {
        "Type mode": type,
        "Reward Config": reward_config,
        "Robot Config": robot_config.__dict__,
        "Sensor Config": sensor_config.__dict__,
        "Env Config": env_config.__dict__,
        "Render Config": render_config.__dict__,
        "Probe Config": probe_config.__dict__,
    }

    print_config_table(config_dict)

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        use_render=False,
        mode=env_config.type,
        type_reward=reward_config,
    )

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
            rewards.append(reward)

            if terminated or truncated:
                obs, _ = env.reset()

    else:
        env = make_vect_envs_norm(
            num_envs=num_envs,
            robot_config=robot_config,
            sensor_config=sensor_config,
            env_config=env_config,
            render_config=render_config,
            use_render=False,
            mode=env_config.type,
            type_reward=reward_config,
        )

        obs, info = env.reset()
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
                actions = [env.action_space.sample() for _ in range(num_envs)]

            obs, rewards, terminated, truncated, infos = env.step(actions)
            ep_rewards += np.array(rewards)
            ep_lengths += 1

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

        # completed_rewards = np.array(completed_rewards)
        # completed_lengths = np.array(completed_lengths)

        # steps_range = list(range(1, len(total_rewards) + 1))
        # step_metrics = [
        #     ("Obstacles Score", obstacles_scores, "brown"),
        #     ("Orientation Score", orientation_scores, "green"),
        #     ("Progress Score", progress_scores, "blue"),
        #     ("Time Score", time_scores, "orange"),
        #     # ("Total Reward", total_rewards, "purple"),
        #     # ("Action", actions_list, "blue"),
        #     ("Distance", dists_list, "cyan"),
        #     ("Alpha", alphas_list, "magenta"),
        #     ("Min Lidar", min_lidars_list, "yellow"),
        #     # ("Max Lidar", max_lidars_list, "black"),
        # ]

        # total_subplots = len(step_metrics) + 1
        # cols = 4
        # rows = (total_subplots + cols - 1) // cols  # 6

        # plt.figure(figsize=(10, 5 * rows))

        # for idx, (title, data, color) in enumerate(step_metrics, start=1):
        #     ax = plt.subplot(rows, cols, idx)
        #     ax.plot(
        #         steps_range,
        #         data,
        #         label=title,
        #         color=color,
        #         linestyle="-",
        #         linewidth=1.5,
        #     )
        #     ax.set_ylabel(title, fontsize=8)
        #     ax.legend(fontsize=6)
        #     ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        #     ax.tick_params(axis="x", labelsize=6)
        #     ax.tick_params(axis="y", labelsize=6)

        #     mean_val = np.mean(data)
        #     min_val = np.min(data)
        #     max_val = np.max(data)

        #     ax.text(
        #         0.5,
        #         -0.25,
        #         f"Média: {mean_val:.4f} | Mínimo: {min_val:.4f} | Máximo: {max_val:.4f}",
        #         transform=ax.transAxes,
        #         ha="center",
        #         fontsize=6,
        #     )

        # ax_ep = plt.subplot(rows, cols, total_subplots)
        # episodes_range = range(1, len(completed_rewards) + 1)

        # ax_ep.plot(
        #     episodes_range, completed_rewards, label="Completed Rewards", color="black"
        # )
        # ax_ep.plot(
        #     episodes_range, completed_lengths, label="Completed Lengths", color="gray"
        # )

        # ax_ep.set_ylabel("Geral", fontsize=8)
        # ax_ep.legend(fontsize=6)
        # ax_ep.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        # ax_ep.tick_params(axis="x", labelsize=6)
        # ax_ep.tick_params(axis="y", labelsize=6)

        # mean_rewards = np.mean(completed_rewards)
        # min_rewards = np.min(completed_rewards)
        # max_rewards = np.max(completed_rewards)

        # mean_lengths = np.mean(completed_lengths)
        # min_lengths = np.min(completed_lengths)
        # max_lengths = np.max(completed_lengths)

        # ax_ep.text(
        #     0.5,
        #     -0.4,
        #     f"Rewards -> Média: {mean_rewards:.4f} | Mínimo: {min_rewards:.4f} | Máximo: {max_rewards:.4f}\n"
        #     f"Lengths -> Média: {mean_lengths:.4f} | Mínimo: {min_lengths:.4f} | Máximo: {max_lengths:.4f}",
        #     transform=ax_ep.transAxes,
        #     ha="center",
        #     fontsize=6,
        # )

        # if image:
        #     plt.tight_layout()
        #     plt.savefig("probe.png", dpi=500, bbox_inches="tight")
        # else:
        #     plt.tight_layout()
        #     plt.subplots_adjust(bottom=0.1)
        #     plt.show()

        completed_rewards = np.array(completed_rewards)
        completed_lengths = np.array(completed_lengths)
        steps_range = list(range(1, len(total_rewards) + 1))

        # Lista de métricas (7 plots) e o 8º será o de episódios
        step_metrics = [
            ("Obstacles Score", obstacles_scores, "brown"),
            ("Orientation Score", orientation_scores, "green"),
            ("Progress Score", progress_scores, "blue"),
            ("Total Reward", total_rewards, "purple"),
            ("Distance", dists_list, "cyan"),
            ("Alpha", alphas_list, "magenta"),
            ("Min Lidar", min_lidars_list, "yellow"),
        ]

        # Cria uma grade de 2 linhas x 4 colunas (8 subplots)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        # Preenche os 7 subplots com as métricas
        for i, (title, data, color) in enumerate(step_metrics):
            ax = axes[i]
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

        # Último subplot: gráfico de episódios (recompensa e tempo médio)
        ax_ep = axes[-1]
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

        print(image)

        if image:
            plt.savefig("probe.png", dpi=500, bbox_inches="tight")
        else:
            plt.subplots_adjust(bottom=0.1)
            plt.show()
