from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from rnl.environment.env import NaviEnv
from training.interface import make, render, robot, sensor
from torch import nn


def make_env():
    param_robot = robot(
        base_radius=0.105,
        vel_linear=[0.0, 0.22],
        vel_angular=[1.0, 2.84],
        wheel_distance=0.16,
        weight=1.0,
        threshold=1.0,
        collision=1.0,
        path_model="",
    )

    # 2.step -> config sensors [for now only lidar sensor!!]
    param_sensor = sensor(
        fov=270,
        num_rays=5,
        min_range=1.0,
        max_range=90.0,
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
    with wandb.init(sync_tensorboard=True, monitor_gym=True) as run:
        config = run.config
        env = DummyVecEnv([make_env])
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
            seed=42,
            verbose=1,
            tensorboard_log=f"runs/{run.id}"
        )
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=WandbCallback(
                gradient_save_freq=10,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
        )

sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "rollout/ep_rew_mean"},
    "parameters": {
        "learning_rate": {"min": 1e-5, "max": 1e-3},
        "batch_size": {"values": [64, 128]},
        "gamma": {"min": 0.9, "max": 0.9999},
        "ent_coef": {"min": 0.0, "max": 0.1},
        "total_timesteps": {"value": 1000},
        "activation_fn": {"values": ["ReLU", "LeakyReLU"]},
        "pi_layers": {"values": [[128, 128], [256, 256]]},
        "vf_layers": {"values": [[128, 128], [256, 256]]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="sb3_sweep")
wandb.agent(sweep_id, function=train, count=2)
