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
    noise: bool
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
    max_vel_linear: float
    max_vel_angular: float
    wheel_distance: float
    weight: float
    threshold: float
    collision: float
    path_model: str

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
    clip_range_vf: float
    target_kl: float
    name: str
    verbose: bool
    policy: str


@dataclass
class ProbeEnvConfig:
    num_envs: int
    max_steps: int
