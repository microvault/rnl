from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from wandb.integration.sb3 import WandbCallback
from rnl.environment.env import NaviEnv
from rnl.training.interface import make, render, robot, sensor
from torch import nn
from sb3_contrib import RecurrentPPO


def make_env():
    param_robot = robot(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=1.0,
        collision=0.5,
        path_model="",
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = sensor(
        fov=270,
        num_rays=5,
        min_range=1.0,
        max_range=12.0,
    )

    # 3.step -> config env
    param_env = make(
        folder_map="./data/map4",
        name_map="map4",
        max_timestep=1000,
        mode="easy-01",
    )

    # 4.step -> config render
    param_render = render(controller=False, debug=False, plot=False)

    env = NaviEnv(
        param_robot, param_sensor, param_env, param_render, use_render=False
    )
    env = Monitor(env)
    return env

def train():
    print("Inicializing training...")
    with wandb.init(sync_tensorboard=True, monitor_gym=True, save_code=False) as run:
        config = run.config
        print(config)

        env = DummyVecEnv([make_env])
        if config.algorithm == "PPO":
            print("Using PPO")
            activation_fn_map = {
                "ReLU": nn.ReLU,
                "LeakyReLU": nn.LeakyReLU,
            }
            activation_fn = activation_fn_map[config.activation_fn]

            policy_kwargs = dict(
                activation_fn=activation_fn,
                net_arch=dict(pi=config.pi_layers, vf=config.vf_layers)
            )

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                gamma=config.gamma,
                ent_coef=config.ent_coef,
                policy_kwargs=policy_kwargs,
                seed=config.seed,
                verbose=1,
                device=config.device,
                tensorboard_log=f"runs/{run.id}"
            )
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=WandbCallback(
                    gradient_save_freq=10,
                    model_save_path=f"models_normal_ppo/{run.id}",
                    verbose=2,
                ),
            )
        elif config.algorithm == "RecurrentPPO":
            print("Using RecurrentPPO")
            activation_fn_map = {
                "ReLU": nn.ReLU,
                "LeakyReLU": nn.LeakyReLU,
            }
            activation_fn = activation_fn_map[config.activation_fn]

            policy_kwargs = dict(
                activation_fn=activation_fn,
                net_arch=dict(pi=config.pi_layers, vf=config.vf_layers)
            )

            model = RecurrentPPO("MlpLstmPolicy",
                env,
                verbose=1,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                gamma=config.gamma,
                seed=config.seed,
                policy_kwargs=policy_kwargs,
                n_steps=config.n_steps,
                clip_range=config.clip_range,
                target_kl=config.target_kl,
                vf_coef=config.vf_coef,
                ent_coef=config.ent_coef,
                device=config.device,
                tensorboard_log=f"runs/{run.id}"
            )
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=WandbCallback(
                    gradient_save_freq=10,
                    model_save_path=f"models_recurrent_ppo/{run.id}",
                    verbose=2,
                ),
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sweep")
    # Novos argumentos adicionados
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=5000000,
        help="Total timesteps for training (default: 5000000)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        help="Algorithm to use (default: PPO)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of sweep runs (default: 20)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="rnl-hyperparameter-search",
        help="WandB project name (default: rnl-hyperparameter-search)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device (default: cuda:1)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default="42",
        help="Seed (default: 42)"
    )

    args = parser.parse_args()

    sweep_config = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "rollout/ep_rew_mean"},
        "parameters": {
            "track_metrics": {
                "value": ["rollout/ep_len_mean", "train/loss"]
            },
            "algorithm": {"values": [args.algorithm]},
            "learning_rate": {"min": 0.000005, "max": 0.003},
            "batch_size": {"values": [128, 1024]},
            "gamma": {"min": 0.9, "max": 0.9999},
            "n_steps": {"min": 32, "max": 5000},
            "clip_range": {"values": [0.1, 0.2, 0.3]},
            # "target_kl": {"min": 0.003, "max": 0.03},
            "vf_coef": {"values": [0.5, 1]},
            "ent_coef": {"min": 0.0, "max": 0.01},
            "total_timesteps": {"value": args.total_timesteps},
            "activation_fn": {"values": ["ReLU", "LeakyReLU"]},
            "pi_layers": {"values": [[128, 128], [256, 256], [128, 256], [256, 128]]},
            "vf_layers": {"values": [[128, 128], [256, 256], [128, 256], [256, 128]]},
            "seed": {"value": args.seed},
            "device": {"value": args.device}
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=args.project)
    wandb.agent(sweep_id, function=train, count=args.count)
