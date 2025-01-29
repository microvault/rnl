from typing import List, Optional

from rnl.configs.config import (
    EnvConfig,
    NetworkConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.training.learn import inference, probe_envs, training

def robot(
    base_radius: float,
    vel_linear: List,
    vel_angular: List,
    wheel_distance: float,
    weight: float,
    threshold: float,
    collision: float,
    path_model: str,
) -> RobotConfig:
    return RobotConfig(
        base_radius,
        vel_linear,
        vel_angular,
        wheel_distance,
        weight,
        threshold,
        collision,
        path_model,
    )


def sensor(
    fov: float, num_rays: int, min_range: float, max_range: float
) -> SensorConfig:
    return SensorConfig(fov, num_rays, min_range, max_range)


def make(
    folder_map: str,
    name_map: str,
    max_timestep: int,
    mode: str,
) -> EnvConfig:
    return EnvConfig(
        folder_map=folder_map,
        name_map=name_map,
        timestep=max_timestep,
        mode=mode,
    )


def render(controller: bool, debug: bool, plot: bool) -> RenderConfig:
    return RenderConfig(controller, debug, plot)


class Trainer:
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
    ):
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config

    def learn(
        self,
        algorithm: str,
        max_timestep_global: int,
        seed: int,
        hidden_size: List[int],
        batch_size: int
        num_envs: int,
        device: str,
        activation: str,
        checkpoint: int,
        use_wandb: bool,
        wandb_api_key: str,
    ) -> None:

        network_config = NetworkConfig(
             hidden_size=hidden_size,
             mlp_activation=activation,

        )
        trainer_config = TrainerConfig(
            algorithm=algorithm,
            max_timestep_global=max_timestep_global,
            seed=seed,
            hidden_size=hidden_size,
            batch_size=batch_size,
            num_envs=num_envs,
            device=device,
            checkpoint=checkpoint,
            use_wandb=use_wandb,
            wandb_api_key=wandb_api_key,
        )
            
        training(
            trainer_config,
            network_config,
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
        )

        return None

class Simulation:
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
        pretrained_model: bool,
    ) -> None:
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config
        self.pretrained_model = pretrained_model

    def run(self) -> None:

        inference(
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
            self.pretrained_model,
        )

        return None


class Probe:
    def __init__(
        self,
        csv_file: str,
        num_envs: int,
        max_steps: int,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
        pretrained_model: bool,
    ) -> None:

        self.csv_file = csv_file
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config
        self.pretrained_model = pretrained_model

    def execute(self) -> None:

        probe_envs(
            self.csv_file,
            self.num_envs,
            self.max_steps,
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
            self.pretrained_model,
        )

        return None
