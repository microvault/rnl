from dataclasses import dataclass
from typing import List


@dataclass
class EnvConfig:
    scalar: int
    grid_length: float
    folder_map: str
    name_map: str
    timestep: int


@dataclass
class RenderConfig:
    controller: bool
    debug: bool
    plot: bool


@dataclass
class SensorConfig:
    fov: float
    num_rays: int
    min_range: float
    max_range: float


@dataclass
class RobotConfig:
    base_radius: float
    vel_linear: List[float]
    vel_angular: List[float]
    wheel_distance: float
    weight: float
    threshold: float
    collision: float
    path_model: str


@dataclass
class NetworkConfig:
    hidden_size: List[int]
    mlp_activation: str


@dataclass
class TrainerConfig:
    use_agents: bool
    max_timestep_global: int
    seed: int
    batch_size: int
    lr: float
    num_envs: int
    device: str
    learn_step: int
    checkpoint: int
    checkpoint_path: str
    use_wandb: bool
    wandb_api_key: str
    llm_api_key: str
    gae_lambda: float
    action_std_init: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    update_epochs: int
    name: str
    save_path: str
    elite_path: str
    overwrite_checkpoints: bool
    save_elite: bool
    evo_steps: int


@dataclass
class ProbeEnvConfig:
    num_envs: int
    max_steps: int
