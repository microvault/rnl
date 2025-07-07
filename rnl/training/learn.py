from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import os
import random
import wandb
from rnl.network.model import CustomActorCriticPolicy
from rnl.agents.evaluate import evaluate_agent, statistics
from rnl.engine.utils import print_config_table
from rnl.configs.config import (
    EnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.configs.rewards import RewardConfig
from rnl.environment.env import NaviEnv
from rnl.training.callback import DynamicTrainingCallback

def training(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    trainer_config: TrainerConfig,
    reward_config: RewardConfig,
    print_parameter: bool,
):

    extra_info = {
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
    verbose_value = 0 if not trainer_config.verbose else 1
    model = None

    def make_env():
        env = NaviEnv(
            robot_config,
            sensor_config,
            env_config,
            render_config,
            use_render=False,
            type_reward=reward_config,
        )
        env = Monitor(env)
        return env

    vec_env = make_vec_env(make_env, n_envs=trainer_config.num_envs)

    policy_kwargs = dict(hidden_sizes=(trainer_config.hidden_size, trainer_config.hidden_size, 64))


    if trainer_config.pretrained != "None":
        model = PPO.load(trainer_config.pretrained)
    else:
        model = PPO(
            policy=CustomActorCriticPolicy,
            env=vec_env,
            batch_size=trainer_config.batch_size,
            policy_kwargs=policy_kwargs,
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
        type_reward=reward_config,
    )

    env.render()
