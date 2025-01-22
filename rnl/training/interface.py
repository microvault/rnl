from typing import List, Optional

from rnl.configs.config import (
    EnvConfig,
    HPOConfig,
    NetworkConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.training.learn import inference, probe_envs, training, learn_with_sb3


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


def render(controller: bool, debug: bool) -> RenderConfig:
    return RenderConfig(controller, debug)


class Trainer:
    def __init__(
        self,
        robot_config: RobotConfig,
        sensor_config: SensorConfig,
        env_config: EnvConfig,
        render_config: RenderConfig,
        pretrained_model: bool,
        train_docker: bool,
        probe: bool,
    ):
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.env_config = env_config
        self.render_config = render_config
        self.pretrained_model = pretrained_model
        self.train_docker = train_docker
        self.probe = probe

    def learn(
        self,
        max_timestep_global: int,
        gamma: float,
        batch_size: int,
        lr: float,
        num_envs: int,
        device: str,
        learn_step: int,
        checkpoint: int,
        checkpoint_path: str,
        overwrite_checkpoints: bool,
        use_mutation: bool,
        population_size: int,
        no_mutation: float,
        arch_mutation: float,
        new_layer: float,
        param_mutation: float,
        active_mutation: float,
        hp_mutation: float,
        hp_mutation_selection: list,
        mutation_strength: float,
        save_elite: bool,
        elite_path: str,
        tourn_size: int,
        elitism: bool,
        hidden_size: list,
        use_wandb: bool,
        wandb_api_key: str,
        min_lr: float,
        max_lr: float,
        min_learn_step: int,
        max_learn_step: int,
        min_batch_size: int,
        max_batch_size: int,
        evo_steps: int,
        eval_loop: int,
        mutate_elite: bool,
        rand_seed: int,
        activation: List,
        mlp_activation: str,
        mlp_output_activation: str,
        min_hidden_layers: int,
        max_hidden_layers: int,
        min_mlp_nodes: int,
        max_mlp_nodes: int,
        gae_lambda: float,
        action_std_init: float,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        update_epochs: int,
        eval_steps: Optional[int] = None,
    ) -> None:

        if use_mutation:

            import matplotlib

            matplotlib.use("Agg")

            trainer_config = TrainerConfig(
                max_timestep_global=max_timestep_global,
                gamma=gamma,
                batch_size=batch_size,
                lr=lr,
                num_envs=num_envs,
                device=device,
                learn_step=learn_step,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                overwrite_checkpoints=overwrite_checkpoints,
                use_wandb=use_wandb,
                wandb_api_key=wandb_api_key,
                gae_lambda=gae_lambda,
                action_std_init=action_std_init,
                clip_coef=clip_coef,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                update_epochs=update_epochs,
            )

            hpo_config = HPOConfig(
                use_mutation=use_mutation,
                population_size=population_size,
                no_mutation=no_mutation,
                arch_mutation=arch_mutation,
                new_layer=new_layer,
                param_mutation=param_mutation,
                active_mutation=active_mutation,
                hp_mutation=hp_mutation,
                hp_mutation_selection=hp_mutation_selection,
                mutation_strength=mutation_strength,
                min_lr=min_lr,
                max_lr=max_lr,
                min_learn_step=min_learn_step,
                max_learn_step=max_learn_step,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                save_elite=save_elite,
                elite_path=elite_path,
                tourn_size=tourn_size,
                elitism=elitism,
                evo_steps=evo_steps,
                eval_steps=eval_steps,
                eval_loop=eval_loop,
                mutate_elite=mutate_elite,
                rand_seed=rand_seed,
                activation=activation,
            )

            network_config = NetworkConfig(
                arch="mlp",
                hidden_size=hidden_size,
                mlp_activation=mlp_activation,
                mlp_output_activation=mlp_output_activation,
                min_hidden_layers=min_hidden_layers,
                max_hidden_layers=max_hidden_layers,
                min_mlp_nodes=min_mlp_nodes,
                max_mlp_nodes=max_mlp_nodes,
            )

            training(
                trainer_config,
                hpo_config,
                network_config,
                self.robot_config,
                self.sensor_config,
                self.env_config,
                self.render_config,
                self.pretrained_model,
                self.train_docker,
                self.probe,
            )

            return None

        else:
            learn_with_sb3(
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
