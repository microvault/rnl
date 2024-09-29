from typing import List

from rnl.configs.config import (
    AgentConfig,
    EnvConfig,
    HPOConfig,
    NetworkConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.training.learn import inference, training


def robot(
    base_radius: float,
    vel_linear: List,
    val_angular: List,
    wheel_distance: float,
    weight: float,
    threshold: float,
):
    return RobotConfig(
        base_radius, vel_linear, val_angular, wheel_distance, weight, threshold
    )


def sensor(fov: float, num_rays: int, min_range: float, max_range: float):
    return SensorConfig(fov, num_rays, min_range, max_range)


def make(
    map_file: str,
    random_mode: str,
    timestep: int,
    grid_dimension: int,
    friction: float,
    porcentage_obstacles: float,
    max_step: int = 1000,
):
    return EnvConfig(
        map_file,
        random_mode,
        timestep,
        grid_dimension,
        friction,
        porcentage_obstacles,
        max_step,
    )


def render(fps: int, controller: bool, rgb_array: bool):
    return RenderConfig(fps, controller, rgb_array)


class Trainer:
    def __init__(
        self,
        robot_config: RobotConfig = RobotConfig(),
        sensor_config: SensorConfig = SensorConfig(),
        env_config: EnvConfig = EnvConfig(),
        render_config: RenderConfig = RenderConfig(),
        pretrained_model=False,
    ):
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config
        self.pretrained_model = pretrained_model

    def learn(
        self,
        max_timestep: int = 800000,
        memory_size: int = 1000000,
        gamma: float = 0.99,
        n_step: int = 3,
        alpha: float = 0.6,
        beta: float = 0.4,
        tau: float = 0.001,
        prior_eps: float = 0.000001,
        num_atoms: int = 51,
        v_min: int = -200,
        v_max: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,  # Batch size
        lr: float = 0.0001,  # Learning rate
        seed: int = 1,
        num_envs: int = 16,
        device: str = "mps",
        learn_step: int = 10,
        target_score: int = 200,
        max_steps: int = 1000000,
        evaluation_steps: int = 10000,
        evaluation_loop: int = 1,
        learning_delay: int = 0,
        n_step_memory: int = 1,
        checkpoint: int = 100,
        checkpoint_path: str = "checkpoints",
        overwrite_checkpoints: bool = False,
        use_wandb: bool = False,
        wandb_api_key: str = "",
        accelerator: bool = False,
        use_mutation: bool = True,
        freq_evolution: int = 10000,
        population_size: int = 6,
        no_mutation: float = 0.4,
        arch_mutation: float = 0.2,
        new_layer: float = 0.2,
        param_mutation: float = 0.2,
        active_mutation: float = 0,
        hp_mutation: float = 0.2,
        hp_mutation_selection: list = ["lr", "batch_size"],
        mutation_strength: float = 0.1,
        evolution_steps: int = 10000,
        save_elite: bool = False,
        elite_path: str = "elite",
        tourn_size: int = 2,
        elitism: bool = True,
        hidden_size: list = [800, 600],
    ):
        agent_config = AgentConfig(
            max_timestep,
            memory_size,
            gamma,
            n_step,
            alpha,
            beta,
            tau,
            prior_eps,
            num_atoms,
            v_min,
            v_max,
            epsilon_start,
            epsilon_end,
            epsilon_decay,
        )

        trainer_config = TrainerConfig(
            batch_size,
            lr,
            seed,
            num_envs,
            device,
            learn_step,
            target_score,
            max_steps,
            evaluation_steps,
            evaluation_loop,
            learning_delay,
            n_step_memory,
            checkpoint,
            checkpoint_path,
            overwrite_checkpoints,
            use_wandb,
            wandb_api_key,
            accelerator,
        )

        hpo_config = HPOConfig(
            use_mutation,
            freq_evolution,
            population_size,
            no_mutation,
            arch_mutation,
            new_layer,
            param_mutation,
            active_mutation,
            hp_mutation,
            hp_mutation_selection,
            mutation_strength,
            evolution_steps,
            save_elite,
            elite_path,
            tourn_size,
            elitism,
        )

        network_config = NetworkConfig(hidden_size)

        training(
            agent_config,
            trainer_config,
            hpo_config,
            network_config,
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
            self.pretrained_model,
        )

    def run(self):

        inference(
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
            self.pretrained_model,
        )
