from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EnvConfig:
    folder_map: str
    name_map: str
    timestep: int
    mode: str


@dataclass
class CurriculumTargetPositionConfig:
    total_steps: int
    min_fraction: float
    max_fraction: float
    increase_smoothness: float


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
    arch: str
    hidden_size: List[int]
    mlp_activation: str
    mlp_output_activation: str
    min_hidden_layers: int
    max_hidden_layers: int
    min_mlp_nodes: int
    max_mlp_nodes: int


@dataclass
class TrainerConfig:
    algorithms: str
    buffer_size: int
    max_timestep_global: int
    gamma: float
    seed: int
    batch_size: int
    lr: float
    num_envs: int
    device: str
    learn_step: int
    checkpoint: int
    checkpoint_path: str
    overwrite_checkpoints: bool
    use_wandb: bool
    wandb_api_key: str
    gae_lambda: float
    action_std_init: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    update_epochs: int
