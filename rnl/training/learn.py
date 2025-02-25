import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from torch import nn
from tqdm import trange
from wandb.integration.sb3 import WandbCallback

import wandb
from rnl.algorithms.ppos import Agent
from rnl.configs.config import (
    EnvConfig,
    NetworkConfig,
    ProbeEnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.engine.utils import plot_metrics
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
    num_envs, max_steps, robot_config, sensor_config, env_config, render_config
):
    assert num_envs >= 1, "num_envs must be greater than 1"

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

    # Cria e checa o ambiente
    env = NaviEnv(
        robot_config, sensor_config, env_config, render_config, use_render=False
    )
    print("\nCheck environment ...")
    check_env(env)

    N = 20
    batch_size = 1024
    n_epochs = 1000
    alpha = 0.0003
    agent = Agent(
        n_actions=env.action_space.n,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape,
    )
    n_games = 1000
    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            # agent.save_models()

        print(
            "episode",
            i,
            "score %.1f" % score,
            "avg score %.1f" % avg_score,
            "time_steps",
            n_steps,
            "learning_steps",
            learn_iters,
        )
    # env = make_vect_envs(
    #     num_envs=num_envs,
    #     robot_config=robot_config,
    #     sensor_config=sensor_config,
    #     env_config=env_config,
    #     render_config=render_config,
    #     use_render=False,
    # )
    # obs, _ = env.reset()

    # # Inicializa variáveis
    # ep_rewards = np.zeros(num_envs)
    # ep_lengths = np.zeros(num_envs)
    # completed_rewards, completed_lengths = [], []
    # metrics = {
    #     "obstacles": [],
    #     "collision": [],
    #     "orientation": [],
    #     "progress": [],
    #     "time": [],
    #     "total_reward": [],
    #     "action": [],
    #     "distance": [],
    #     "alpha": [],
    #     "min_lidar": [],
    #     "max_lidar": [],
    #     "steps_to_goal": [],
    #     "steps_below_threshold": [],
    #     "steps_to_collision": [],
    #     "turn_left_count": [],
    #     "turn_right_count": [],
    # }

    # # Carrega o modelo se existir
    # if robot_config.path_model != "None":
    #     model = PPO.load(robot_config.path_model)

    # pbar = trange(max_steps, desc="Probe envs", unit="step")
    # for _ in pbar:
    #     # Obtém ação
    #     if robot_config.path_model != "None":
    #         actions, _ = model.predict(obs)
    #     else:
    #         actions = env.action_space.sample()

    #     obs, rewards, terminated, truncated, infos = env.step(actions)
    #     ep_rewards += np.array(rewards)
    #     ep_lengths += 1

    #     # Atualiza total_reward (acumula a média dos rewards do step)
    #     metrics["total_reward"].append(np.mean(rewards))

    #     # Atualiza outros métricos, filtrando valores None
    #     for key in ["obstacles", "collision", "orientation", "progress", "time",
    #                 "action", "distance", "alpha", "min_lidar", "max_lidar",
    #                 "steps_to_goal", "steps_below_threshold", "steps_to_collision",
    #                 "turn_left_count", "turn_right_count"]:
    #         value = infos.get(key, [0.0] * num_envs)
    #         clean_value = [v if v is not None else 0.0 for v in value]
    #         metrics[key].append(np.mean(clean_value))

    #     # Verifica término de episódio
    #     done = np.logical_or(terminated, truncated)
    #     done_indices = np.where(done)[0]
    #     if done_indices.size > 0:
    #         for idx in done_indices:
    #             completed_rewards.append(ep_rewards[idx])
    #             completed_lengths.append(ep_lengths[idx])
    #             ep_rewards[idx] = 0
    #             ep_lengths[idx] = 0

    #     # Atualiza o progresso no pbar
    #     if completed_rewards:
    #         avg_reward = np.mean(completed_rewards[-100:])
    #         avg_length = np.mean(completed_lengths[-100:])
    #     else:
    #         avg_reward = avg_length = 0
    #     pbar.set_postfix({
    #         "Ep Comp.": len(completed_rewards),
    #         "Mean Reward(100)": f"{avg_reward:.2f}",
    #         "Mean Length(100)": f"{avg_length:.2f}",
    #     })

    # # Finaliza episódios que ainda estão ativos
    # for idx in range(num_envs):
    #     if ep_lengths[idx] > 0:
    #         completed_rewards.append(ep_rewards[idx])
    #         completed_lengths.append(ep_lengths[idx])
    # env.close()

    # if render_config.debug:
    #     plot_metrics(metrics, completed_rewards, completed_lengths)
