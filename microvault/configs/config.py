from dataclasses import dataclass


@dataclass
class WandbConfig:
    project: str
    entity: str
    mode: str
    use_sweep: bool


@dataclass
class AgentConfig:
    gamma: float
    tau: float
    update_every_step: int
    lr_model: float
    weight_decay: float


@dataclass
class ReplayBufferConfig:
    buffer_size: int


@dataclass
class EnvironmentConfig:
    timestep: int
    size: float
    fps: int
    random: int
    threshold: float
    grid_lenght: int
    state_size: int
    action_size: int


@dataclass
class RobotConfig:
    wheel_radius: float
    wheel_base: float
    fov: float
    num_rays: int
    max_range: float


@dataclass
class EngineConfig:
    seed: int
    device: str
    batch_size: int
    pretrained: bool
    path: str
    wandb: bool
    save_checkpoint: int
    epochs: int


@dataclass
class NetworkConfig:
    layers_model_l1: int
    layers_model_l2: int


@dataclass
class TrainerConfig:
    wandb: WandbConfig
    agent: AgentConfig
    replay_buffer: ReplayBufferConfig
    environment: EnvironmentConfig
    engine: EngineConfig
    network: NetworkConfig
    robot: RobotConfig
