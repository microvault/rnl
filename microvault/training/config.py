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
    lr_actor: float
    lr_critic: float
    noise: float
    noise_std: float
    noise_clip: float
    normal_scalar: float
    nstep: int
    eta: float
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
    min_radius: float
    max_radius: float
    wheel_radius: float
    wheel_base: float
    fov: float
    num_rays: int
    max_range: float
    max_linear: float
    min_linear: float
    max_angular: float
    min_angular: float
    max_action: float
    min_action: float
    size: float


@dataclass
class EngineConfig:
    seed: int
    device: str
    batch_size: int
    num_episodes: int
    pretrained: bool
    checkpoint: int
    path: str
    wandb: bool
    save_checkpoint: int
    epochs: int


@dataclass
class NetworkConfig:
    layers_actor_l1: int
    layers_actor_l2: int
    layers_critic_l1: int
    layers_critic_l2: int
    activation: str
    noise_std: float


@dataclass
class TrainerConfig:
    wandb: WandbConfig
    agent: AgentConfig
    replay_buffer: ReplayBufferConfig
    environment: EnvironmentConfig
    engine: EngineConfig
    network: NetworkConfig
    robot: RobotConfig
