from typing import List

from rnl.configs.config import (
    EnvConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.configs.rewards import RewardConfig
from rnl.training.learn import inference, probe_envs, training
from rnl.training.train import run_multiple_parallel_trainings

def robot(
    base_radius: float,
    max_vel_linear: float,
    max_vel_angular: float,
    wheel_distance: float,
    weight: float,
    threshold: float,
    collision: float,
    path_model: str,
) -> RobotConfig:

    parameter = {
        "Base radius": base_radius,
        "Wheel distance": wheel_distance,
        "Weight": weight,
    }

    for name, value in parameter.items():
        if value <= 0:
            raise ValueError(f"Error: {name} must be greater than 0.")

    if threshold < 0:
        raise ValueError("Error: Threshold must be greater than or equal to 0.")
    if collision < 0:
        raise ValueError("Error: Collision must be greater than or equal to 0.")
    if path_model == "":
        path_model = "None"

    return RobotConfig(
        base_radius,
        max_vel_linear,
        max_vel_angular,
        wheel_distance,
        weight,
        threshold,
        collision,
        path_model,
    )

def sensor(
    fov: float, num_rays: int, min_range: float, max_range: float
) -> SensorConfig:
    if min_range < 0 or max_range < 0:
        raise ValueError(
            "Error: Minimum/Maximum range must be greater than or equal to 0."
        )
    if num_rays < 0:
        raise ValueError("Error: Number of rays must be greater than or equal to 0.")
    if fov < 0 or fov > 360:
        raise ValueError("Error: Field of view must be between 0 and 360 degrees.")
    if min_range > max_range:
        raise ValueError(
            "Error: Minimum range must be less than or equal to maximum range."
        )
    if min_range == max_range:
        raise ValueError("Error: Minimum range cannot be equal to maximum range.")
    if num_rays == 0:
        raise ValueError("Error: Number of rays cannot be equal to 0.")
    if fov == 0:
        raise ValueError("Error: Field of view cannot be equal to 0.")
    if num_rays > 10:
        raise ValueError("Error: Number of rays must be less than or equal to 10.")

    return SensorConfig(fov, num_rays, min_range, max_range)


def make(
    scalar: int,
    folder_map: str,
    name_map: str,
    max_timestep: int,
    type: str,
    grid_size: List,
    map_size: int,
    noise: bool,
    obstacle_percentage: float
) -> EnvConfig:

    if scalar < 0 or scalar > 100:
        raise ValueError("Error: Scalar must be between 0 and 100.")
    if max_timestep < 0:
        raise ValueError("Error: Maximum timestep must be greater than 0.")

    return EnvConfig(
        scalar=scalar,
        folder_map=folder_map,
        name_map=name_map,
        timestep=max_timestep,
        obstacle_percentage=obstacle_percentage,
        map_size=map_size,
        type=type,
        noise=noise,
        grid_size=grid_size,
    )


def render(controller: bool, debug: bool, plot: bool = False) -> RenderConfig:
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
        population: int,
        loop_feedback: int,
        description_task: str,
        pretrained: str,
        use_agents: bool,
        max_timestep_global: int,
        seed: int,
        batch_size: int,
        num_envs: int,
        device: str,
        checkpoint: int,
        checkpoint_path: str,
        use_wandb: bool,
        wandb_api_key: str,
        llm_api_key: str,
        lr: float,
        learn_step: int,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        update_epochs: int,
        clip_range_vf: float,
        target_kl: float,
        name: str,
        verbose: bool,
        policy: str,
    ):

        if seed < 0:
            raise ValueError("Error: Seed must be greater than or equal to 0.")
        if max_timestep_global < 0:
            raise ValueError("Error: Maximum timestep must be greater than 0.")
        if batch_size < 0:
            raise ValueError("Error: Batch size must be greater than 0.")
        if num_envs < 0:
            raise ValueError("Error: Number of environments must be greater than 0.")
        if learn_step < 0:
            raise ValueError("Error: Learning step must be greater than 0.")
        if gae_lambda < 0:
            raise ValueError("Error: GAE lambda must be greater than 0.")
        if ent_coef < 0:
            raise ValueError("Error: Entropy coefficient must be greater than 0.")
        if vf_coef < 0:
            raise ValueError(
                "Error: Value function coefficient must be greater than 0."
            )
        if max_grad_norm < 0:
            raise ValueError("Error: Maximum gradient norm must be greater than 0.")
        if update_epochs < 0:
            raise ValueError("Error: Update epochs must be greater than 0.")

        trainer_config = TrainerConfig(
            pretrained=pretrained,
            use_agents=use_agents,
            max_timestep_global=max_timestep_global,
            seed=seed,
            batch_size=batch_size,
            num_envs=num_envs,
            device=device,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            use_wandb=use_wandb,
            wandb_api_key=wandb_api_key,
            wandb_mode="online",
            llm_api_key=llm_api_key,
            lr=lr,
            learn_step=learn_step,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            update_epochs=update_epochs,
            clip_range_vf=clip_range_vf,
            target_kl=target_kl,
            name=name,
            verbose=verbose,
            policy=policy,
        )

        if self.render_config.plot:
            raise ValueError("Error: Plot mode is not supported for training.")
        if self.render_config.controller:
            raise ValueError("Error: Controller mode is not supported for training.")

        reward_config = RewardConfig(
            params={
                "scale_orientation": 0.0,
                "scale_distance": 0.0,
                "scale_time": 0.01,
                "scale_obstacle": 0.0,
                "scale_angular": 0.0,
            },
        )

        if use_agents:
            config = {
                "robot_config": self.robot_config,
                "sensor_config": self.sensor_config,
                "env_config": self.env_config,
                "render_config": self.render_config,
                "trainer_config": trainer_config,
                "network_config": network_config,
                "reward_config": reward_config,
            }
            configs = population * [config]
            run_multiple_parallel_trainings(
                num_loops=loop_feedback,
                initial_configs=configs,
                num_populations=population,
                description_task=description_task
            )
            return None

        else:
            metrics = training(
                self.robot_config,
                self.sensor_config,
                self.env_config,
                self.render_config,
                trainer_config,
                network_config,
                reward_config,
                print_parameter=True,
                train=False,
            )
            return metrics



class Simulation:
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
        type: str
    ) -> None:
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config
        self.type = type

    def run(self) -> None:

        reward_config = RewardConfig(
            params={
                "scale_orientation": 0.0,
                "scale_distance": 0.0,
                "scale_time": 0.01,
                "scale_obstacle": 0.02,
                "scale_angular": 0.005,
            },
        )

        inference(
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
            reward_config,
        )

        return None


class Probe:
    def __init__(
        self,
        seed: int,
        num_envs: int,
        max_steps: int,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
    ) -> None:

        if seed < 0:
            raise ValueError("Error: Seed must be greater than or equal to 0.")
        if max_steps < 0:
            raise ValueError("Error: Maximum steps must be greater than 0.")
        if num_envs < 0:
            raise ValueError("Error: Number of environments must be greater than 0.")

        self.num_envs = num_envs
        self.max_steps = max_steps
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config
        self.seed = seed

    def execute(self) -> None:

        if self.render_config.controller:
            raise ValueError("Error: Controller mode is not supported for training.")

        if self.render_config.plot:
            raise ValueError("Error: Plot mode is not supported for training.")

        reward_config = RewardConfig(
            params={
                "scale_orientation": 0.01,
                "scale_distance": 0.01,
                "scale_time": 0.01,
                "scale_obstacle": 0.01,
                "scale_angular": 0.01,
            },
        )

        probe_envs(
            self.num_envs,
            self.max_steps,
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
            self.seed,
            reward_config
        )

        return None
