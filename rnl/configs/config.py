from dataclasses import dataclass
from typing import List


@dataclass
class EnvConfig:
    scalar: int
    folder_map: str
    name_map: str
    timestep: int
    obstacle_percentage: float
    map_size: float
    type: str
    grid_size: List[float]


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
    noise: bool
    path_model: str


@dataclass
class NetworkConfig:
    hidden_size: List[int]
    mlp_activation: str
    type_model: str


@dataclass
class TrainerConfig:
    pretrained: str
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
    wandb_mode: str
    llm_api_key: str
    gae_lambda: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    update_epochs: int
    name: str
    verbose: bool
    policy_type: str


@dataclass
class ProbeEnvConfig:
    num_envs: int
    max_steps: int
