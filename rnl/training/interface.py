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
    vel_angular: List,
    wheel_distance: float,
    weight: float,
    threshold: float,
    collision: float,
    path_model: str = "",
):
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


def sensor(fov: float, num_rays: int, min_range: float, max_range: float):
    return SensorConfig(fov, num_rays, min_range, max_range)


def make(
    folder_map: str,
    name_map: str,
    max_timestep: int,
):
    return EnvConfig(
        folder_map=folder_map,
        name_map=name_map,
        timestep=max_timestep,
    )


def render(controller: bool):
    return RenderConfig(controller)


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
        max_timestep: int,
        memory_size: int,
        gamma: float,
        n_step: int,
        alpha: float,
        beta: float,
        tau: float,
        prior_eps: float,
        num_atoms: int,
        v_min: int,
        v_max: int,
        epsilon_start: float ,
        epsilon_end: float,
        epsilon_decay: float,
        batch_size: int,
        lr: float,
        seed: int,
        num_envs: int,
        device: str,
        learn_step: int,
        target_score: int,
        max_steps: int,
        evaluation_steps: int,
        evaluation_loop: int,
        learning_delay: int,
        n_step_memory: int,
        checkpoint: int,
        checkpoint_path: str,
        overwrite_checkpoints: bool,
        use_mutation: bool,
        freq_evolution: int,
        population_size: int,
        no_mutation: float,
        arch_mutation: float,
        new_layer: float,
        param_mutation: float,
        active_mutation: float,
        hp_mutation: float,
        hp_mutation_selection: list,
        mutation_strength: float,
        evolution_steps: int,
        save_elite: bool,
        elite_path: str,
        tourn_size: int,
        elitism: bool,
        hidden_size: list,
        use_wandb: bool,
        wandb_api_key: str,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        noise_std: float,
        per: bool,
        min_lr: float,
        max_lr: float,
        min_learn_step: int,
        max_learn_step: int,
        min_batch_size: int,
        max_batch_size: int,
        evo_steps: int,
        eval_steps: int,
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
    ) -> None:

        agent_config = AgentConfig(
            max_timestep=max_timestep,
            memory_size=memory_size,
            gamma=gamma,
            n_step=n_step,
            alpha=alpha,
            beta=beta,
            tau=tau,
            prior_eps=prior_eps,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            noise_std=noise_std,
            per=per,
        )

        trainer_config = TrainerConfig(
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            num_envs=num_envs,
            device=device,
            learn_step=learn_step,
            target_score=target_score,
            max_steps=max_steps,
            learning_delay=learning_delay,
            n_step_memory=n_step_memory,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            overwrite_checkpoints=overwrite_checkpoints,
            use_wandb=use_wandb,
            wandb_api_key=wandb_api_key,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
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

        return None


class Simulation:
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

    def run(self):

        inference(
            self.robot_config,
            self.sensor_config,
            self.env_config,
            self.render_config,
            self.pretrained_model,
        )
