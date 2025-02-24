from typing import List

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
    scalar: int,
    grid_length: float,
    folder_map: str,
    name_map: str,
    max_timestep: int,
    mode: str,
    reward_function: str,
) -> EnvConfig:
    return EnvConfig(
        scalar=scalar,
        grid_length=grid_length,
        folder_map=folder_map,
        name_map=name_map,
        timestep=max_timestep,
        mode=mode,
        reward_function=reward_function,
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
        max_timestep_global: int,
        seed: int,
        hidden_size: List[int],
        batch_size: int,
        num_envs: int,
        device: str,
        activation: str,
        checkpoint: str,
        use_wandb: bool,
        wandb_api_key: str,
        lr: float,
        learn_step: int,
        gae_lambda: float,
        action_std_init: float,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        update_epochs: int,
        name: str,
    ) -> None:

        network_config = NetworkConfig(
            hidden_size=hidden_size,
            mlp_activation=activation,
        )
        trainer_config = TrainerConfig(
            max_timestep_global=max_timestep_global,
            seed=seed,
            batch_size=batch_size,
            num_envs=num_envs,
            device=device,
            checkpoint=checkpoint,
            use_wandb=use_wandb,
            wandb_api_key=wandb_api_key,
            lr=lr,
            learn_step=learn_step,
            gae_lambda=gae_lambda,
            action_std_init=action_std_init,
            clip_coef=clip_coef,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            update_epochs=update_epochs,
            name=name,
        )

        if self.render_config.debug:
            raise ValueError("Error: Debug mode is not supported for training.")
        if self.render_config.plot:
            raise ValueError("Error: Plot mode is not supported for training.")
        if self.render_config.controller:
            raise ValueError("Error: Controller mode is not supported for training.")

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
    ) -> None:
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config

    def run(self) -> None:

        inference(
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
        )

        return None


class Probe:
    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
    ) -> None:

        self.num_envs = num_envs
        self.max_steps = max_steps
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config

    def execute(self) -> None:

        if self.render_config.controller:
            raise ValueError("Error: Controller mode is not supported for training.")

        if self.render_config.plot:
            raise ValueError("Error: Plot mode is not supported for training.")

        probe_envs(
            self.num_envs,
            self.max_steps,
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
        )

        return None
