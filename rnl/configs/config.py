from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    folder_map: str = "None"
    name_map: str = "map"
    timestep: int = 1000


@dataclass
class CurriculumTargetPositionConfig:
    total_steps: int = 40_000_000
    min_fraction: float = 0.01
    max_fraction: float = 1.0
    increase_smoothness: float = 1.0


@dataclass
class RenderConfig:
    controller: bool = False


@dataclass
class SensorConfig:
    fov: float = 360.0
    num_rays: int = 40
    min_range: float = 1.0
    max_range: float = 40.0


@dataclass
class RobotConfig:
    base_radius: float = 0.105
    vel_linear: List[float] = field(default_factory=lambda: [0.0, 0.22])
    vel_angular: List[float] = field(default_factory=lambda: [1.0, 2.84])
    wheel_distance: float = 0.160
    weight: float = 1.0
    threshold: float = 2.0
    collision: float = 0.20
    path_model: str = "./"


@dataclass
class NetworkConfig:
    hidden_size: List[int] = field(default_factory=lambda: [800, 600])
    std_init_noisy_linear: float = 0.5
    mlp_activation: str = "ReLU"
    mlp_output_activation: str = "ReLU"
    min_hidden_layers: int = 2
    max_hidden_layers: int = 4
    min_mlp_nodes: int = 64
    max_mlp_nodes: int = 500
    layer_norm: bool = True
    output_vanish: bool = True
    init_layers: bool = True
    noise_std: float = 0.5


@dataclass
class AgentConfig:
    max_timestep: int = 800000
    memory_size: int = 1000000
    gamma: float = 0.99
    n_step: int = 3
    alpha: float = 0.6
    beta: float = 0.4
    tau: float = 0.001
    prior_eps: float = 0.000001
    num_atoms: int = 51
    v_min: int = -200
    v_max: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    noise_std: float = 0.5


@dataclass
class TrainerConfig:
    batch_size: int = 256
    lr: float = 0.0001
    seed: int = 1
    num_envs: int = 1
    device: str = "cuda"
    learn_step: int = 10
    target_score: int = 200
    max_steps: int = 1000000
    evaluation_steps: int = 10000
    evaluation_loop: int = 10
    learning_delay: int = 0
    n_step_memory: int = 2
    checkpoint: int = 1000
    checkpoint_path: str = "checkpoints"
    overwrite_checkpoints: bool = False
    use_wandb: bool = False
    wandb_api_key: str = ""


@dataclass
class HPOConfig:
    use_mutation: bool = True
    freq_evolution: int = 10000
    population_size: int = 6
    no_mutation: float = 0.4
    arch_mutation: float = 0.2
    new_layer: float = 0.2
    param_mutation: float = 0.2
    active_mutation: float = 0
    hp_mutation: float = 0.2
    hp_mutation_selection: List[str] = field(
        default_factory=lambda: ["lr", "batch_size"]
    )
    mutation_strength: float = 0.1
    evolution_steps: int = 10000
    save_elite: bool = False
    elite_path: str = "elite"
    tourn_size: int = 2
    elitism: bool = True


@dataclass
class RandomizationDomainConfig:
    weight: List[float] = field(default_factory=lambda: [0.01, 500])
    base_radius: List[float] = field(default_factory=lambda: [0.01, 0.9])
    wheel_distance: List[float] = field(default_factory=lambda: [0.1, 1.5])
    threshold: List[float] = field(default_factory=lambda: [0.001, 0.1])
    fov: List[float] = field(default_factory=lambda: [3.14159, 12.56637])
    num_rays: List[int] = field(default_factory=lambda: [5, 60])
    min_max_range: List[int] = field(default_factory=lambda: [1, 10])
    range: List[float] = field(default_factory=lambda: [0.1, 40])
    grid_dimension: List[int] = field(default_factory=lambda: [5, 15])
    porcentage_obstacles: List[float] = field(default_factory=lambda: [0.01, 0.5])
