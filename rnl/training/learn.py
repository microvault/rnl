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
    ProbeEnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.configs.rewards import RewardConfig
from rnl.engine.utils import print_config_table, set_seed
from rnl.engine.vector import make_vect_envs, make_vect_envs_norm, _safe_plot
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
        "scale_angular": 0.005,
    },
)


def training(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    trainer_config: TrainerConfig,
    reward_config: RewardConfig,
    print_parameter: bool,
    train: bool,
):

    extra_info = {
        "Type mode": env_config.type,
        "Type policy": trainer_config.policy,
        "scale_orientation": reward_config.params["scale_orientation"],
        "scale_distance": reward_config.params["scale_distance"],
        "scale_time": reward_config.params["scale_time"],
        "scale_obstacle": reward_config.params["scale_obstacle"],
        "scale_angular": reward_config.params["scale_angular"],
    }

    config_dict = {
        "Trainer Config": trainer_config.__dict__,
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

    policy_kwargs_on_policy_recurrent = None
    policy_kwargs_off_policy = None
    policy_kwargs_on_policy_recurrent = dict(
        activation_fn=nn.LeakyReLU,
        net_arch=dict(
            pi=[32, 32],
            vf=[32, 32],
        ),
        n_lstm_layers=1,
        lstm_hidden_size=64,
    )

    policy_kwargs_off_policy = dict(
        activation_fn=nn.LeakyReLU,
        net_arch=[32, 32],
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
        if trainer_config.policy == "TRPO":
            model = TRPO(
                policy=CustomActorCriticPolicy,
                env=vec_env,
                batch_size=trainer_config.batch_size,
                verbose=verbose_value,
                learning_rate=trainer_config.lr,
                n_steps=trainer_config.learn_step,
                device=trainer_config.device,
                seed=trainer_config.seed,
            )
        if trainer_config.policy == "RecurrentPPO":
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
                clip_range_vf=trainer_config.clip_range_vf,
                seed=trainer_config.seed,
            )
        elif trainer_config.policy == "PPO":
            model = PPO(
                policy=CustomActorCriticPolicy,
                env=vec_env,
                batch_size=trainer_config.batch_size,
                verbose=verbose_value,
                learning_rate=trainer_config.lr,
                n_steps=trainer_config.learn_step,
                vf_coef=trainer_config.vf_coef,
                ent_coef=trainer_config.ent_coef,
                device=trainer_config.device,
                max_grad_norm=trainer_config.max_grad_norm,
                n_epochs=trainer_config.update_epochs,
                clip_range_vf=trainer_config.clip_range_vf,
                target_kl=trainer_config.target_kl,
                seed=trainer_config.seed,
            )
        elif trainer_config.policy == "A2C":
            model = A2C(
                policy=CustomActorCriticPolicy,
                env=vec_env,
                verbose=verbose_value,
                learning_rate=trainer_config.lr,
                n_steps=trainer_config.learn_step,
                gae_lambda=trainer_config.gae_lambda,
                ent_coef=trainer_config.ent_coef,
                vf_coef=trainer_config.vf_coef,
                max_grad_norm=trainer_config.max_grad_norm,
                seed=trainer_config.seed,
                device=trainer_config.device,
            )
        elif trainer_config.policy == "DQN":
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
        elif trainer_config.policy == "QRDQN":
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
        run_id=str(id),
        wandb_run=run,
        save_checkpoint=trainer_config.checkpoint,
        model_save_path=trainer_config.checkpoint_path,
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

    scales = {
        "scale_orientation": reward_config.params["scale_orientation"],
        "scale_distance":   reward_config.params["scale_distance"],
        "scale_time":       reward_config.params["scale_time"],
        "scale_obstacle":   reward_config.params["scale_obstacle"],
        "scale_angular":    reward_config.params["scale_angular"],
    }

    eval_keys = [
        "success_percentage",
        "percentage_unsafe",
        "percentage_angular",
        "ep_mean_length",
        "avg_collision_steps",
        "avg_goal_steps",
    ]

    final_eval_dict = dict(zip(eval_keys, final_eval))

    merged_dict = {**metrics, **final_eval_dict, **scales}

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
        obs = env.reset()
        model = PPO.load(robot_config.path_model) if robot_config.path_model != "None" else None

        # ---- buffers ----------------------------------------------------------
        rewards_history = []
        dist_hist, alpha_hist = [], []
        obstacle_hist = []
        progress_hist, orient_hist, time_hist = [], [], []

        completed_rewards, completed_lengths = [], []
        ep_reward, ep_len = 0.0, 0

        # ---- probe loop -------------------------------------------------------
        pbar = trange(max_steps, desc="Probe single env", unit="step")
        for _ in pbar:
            action = model.predict(obs)[0] if model is not None else env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            # ---- store step metrics -------------------------------------------
            rewards_history.append(float(reward))
            if info:
                obstacle_hist .append(info.get("obstacle_score",     0.0))
                orient_hist   .append(info.get("orientation_score",  0.0))
                progress_hist .append(info.get("progress_score",     0.0))
                time_hist     .append(info.get("time_score",         0.0))
                dist_hist     .append(info.get("dist",               0.0))
                alpha_hist    .append(info.get("alpha",              0.0))

            # ---- track episode stats ------------------------------------------
            ep_reward += float(reward)
            ep_len    += 1

            if term:
                completed_rewards.append(ep_reward)
                completed_lengths.append(ep_len)
                ep_reward, ep_len = 0.0, 0
                obs = env.reset()

        env.close()

        # inclui o último episódio caso não tenha terminado
        if ep_len > 0:
            completed_rewards.append(ep_reward)
            completed_lengths.append(ep_len)

        # ---- plotting ---------------------------------------------------------
        steps_range = np.arange(1, len(rewards_history) + 1)

        step_metrics = [
            ("Obstacles Score",  obstacle_hist,  "brown"),
            ("Orientation Score",orient_hist,    "green"),
            ("Progress Score",   progress_hist,  "blue"),
            ("Total Reward",     rewards_history,"purple"),
            ("Distance",         dist_hist,      "cyan"),
            ("Alpha",            alpha_hist,     "magenta"),
            ("Time Score",       time_hist,      "orange"),
        ]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        # --- 7 métricas por passo ---------------------------------------------
        for i, (title, data, color) in enumerate(step_metrics):
            ax = axes[i]
            ax.plot(steps_range, data, color=color, linewidth=1.5, label=title)
            ax.set_ylabel(title, fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            ax.tick_params(axis="x", labelsize=6)
            ax.tick_params(axis="y", labelsize=6)

            mean_val, min_val, max_val = np.mean(data), np.min(data), np.max(data)
            ax.text(
                0.5, -0.25,
                f"µ {mean_val:.4f} | min {min_val:.4f} | max {max_val:.4f}",
                transform=ax.transAxes, ha="center", fontsize=6
            )

        # --- último subplot: métricas por episódio ----------------------------
        ax_ep = axes[-1]
        ep_range = np.arange(1, len(completed_rewards) + 1)
        ax_ep.plot(ep_range, completed_rewards, color="black", label="Episode Rewards")
        ax_ep.plot(ep_range, completed_lengths, color="gray",  label="Episode Lengths")
        ax_ep.set_ylabel("Episódios", fontsize=8)
        ax_ep.legend(fontsize=6)
        ax_ep.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax_ep.tick_params(axis="x", labelsize=6)
        ax_ep.tick_params(axis="y", labelsize=6)

        if completed_rewards:
            r_mean, r_min, r_max = np.mean(completed_rewards), np.min(completed_rewards), np.max(completed_rewards)
            l_mean, l_min, l_max = np.mean(completed_lengths), np.min(completed_lengths), np.max(completed_lengths)
            ax_ep.text(
                0.5, -0.4,
                f"Rewards → µ {r_mean:.2f} | min {r_min:.2f} | max {r_max:.2f}\n"
                f"Lengths → µ {l_mean:.0f} | min {l_min} | max {l_max}",
                transform=ax_ep.transAxes, ha="center", fontsize=6
            )

        plt.tight_layout()
        plt.show()

    else:
        env = make_vect_envs_norm(
            num_envs=num_envs,
            robot_config=robot_config,
            sensor_config=sensor_config,
            env_config=env_config,
            render_config=render_config,
            use_render=False,
            type_reward=reward_config,
        )

        obs = env.reset()
        ep_lengths = np.zeros(num_envs, dtype=np.int32)
        ep_rewards = np.zeros(num_envs, dtype=np.float32)

        completed_rewards, completed_lengths = [], []
        obstacles_scores, orientation_scores, progress_scores, time_scores = [], [], [], []
        actions_list, dists_list, alphas_list = [], [], []
        min_lidars_list, max_lidars_list = [], []
        total_rewards = []

        model = PPO.load(robot_config.path_model) if robot_config.path_model != "None" else None

        pbar = trange(max_steps, desc="Probe envs", unit="step")

        for _ in pbar:
            actions = model.predict(obs)[0] if model else [env.action_space.sample() for _ in range(num_envs)]
            obs, rewards, dones, infos = env.step(actions)

            rewards = np.asarray(rewards, dtype=np.float32)
            ep_rewards += rewards
            ep_lengths += 1

            # -------- métricas por passo ----------------------------------------
            for env_idx, info in enumerate(infos):
                obstacles_scores .append(info.get("obstacle_score", 0.0))
                orientation_scores.append(info.get("orientation_score", 0.0))
                progress_scores  .append(info.get("progress_score", 0.0))
                time_scores      .append(info.get("time_score", 0.0))
                total_rewards.append(rewards[env_idx])

                actions_list .append(info.get("action", 0.0))
                dists_list   .append(info.get("dist", 0.0))
                alphas_list  .append(info.get("alpha", 0.0))
                min_lidars_list.append(info.get("min_lidar", 0.0))
                max_lidars_list.append(info.get("max_lidar", 0.0))

            # -------- episódios concluídos --------------------------------------
            done_idx = np.where(dones)[0]
            for idx in done_idx:
                completed_rewards.append(ep_rewards[idx])
                completed_lengths.append(ep_lengths[idx])
                ep_rewards[idx] = 0.0
                ep_lengths[idx] = 0

            # -------- progresso no tqdm ----------------------------------------
            if completed_rewards:
                avg_r = np.mean(completed_rewards[-100:])
                avg_l = np.mean(completed_lengths[-100:])
            else:
                avg_r = avg_l = 0
            pbar.set_postfix({"Ep Comp.": len(completed_rewards),
                                "Mean Reward(100)": f"{avg_r:.2f}",
                                "Mean length(100)": f"{avg_l:.2f}"})

        # adiciona episódios inacabados
        for idx in range(num_envs):
            if ep_lengths[idx] > 0:
                completed_rewards.append(ep_rewards[idx])
                completed_lengths.append(ep_lengths[idx])

        env.close()

        # --------------------------------------------------------------------------------
        # plot ----------------------------------------------------------------------------
        # --------------------------------------------------------------------------------
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        metrics = [
            ("Obstacles Score",  obstacles_scores, "brown"),
            ("Orientation Score",orientation_scores,"green"),
            ("Progress Score",   progress_scores,  "blue"),
            ("Total Reward",     total_rewards,    "purple"),
            ("Distance",         dists_list,       "cyan"),
            ("Alpha",            alphas_list,      "magenta"),
            ("Min Lidar",        min_lidars_list,  "yellow"),
        ]

        for ax, (title, data, color) in zip(axes[:-1], metrics):
            _safe_plot(ax, data, color, title)

        # ---- episódios --------------------------------------------------------
        ax_ep = axes[-1]
        if completed_rewards:
            x_ep = range(1, len(completed_rewards) + 1)
            ax_ep.plot(x_ep, completed_rewards, color="black", label="Episode Rewards")
            ax_ep.plot(x_ep, completed_lengths, color="gray",  label="Episode Lengths")
            ax_ep.set_ylabel("Episódios", fontsize=8)
            ax_ep.legend(fontsize=6)
            ax_ep.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            ax_ep.tick_params(axis="x", labelsize=6)
            ax_ep.tick_params(axis="y", labelsize=6)
            rµ, rmin, rmax = np.mean(completed_rewards), np.min(completed_rewards), np.max(completed_rewards)
            lµ, lmin, lmax = np.mean(completed_lengths), np.min(completed_lengths), np.max(completed_lengths)
            ax_ep.text(
                0.5, -0.4,
                f"Rewards → µ {rµ:.2f} | min {rmin:.2f} | max {rmax:.2f}\n"
                f"Lengths → µ {lµ:.0f} | min {lmin} | max {lmax}",
                transform=ax_ep.transAxes, ha="center", fontsize=6,
            )
        else:
            ax_ep.set_visible(False)

        plt.tight_layout()
        if image:
            plt.savefig("probe.png", dpi=500, bbox_inches="tight")
        else:
            plt.show()
