from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    map_file: str = "rnl/data/map/map.pgm"
    random_mode: str = "normal"
    timestep: int = 1000
    grid_dimension: int = 5
    friction: float = 0.4
    porcentage_obstacles: float = 0.1
    max_step: int = 1000


@dataclass
class RenderConfig:
    fps: int = 1
    controller: bool = False
    rgb_array: bool = False


@dataclass
class SensorConfig:
    fov: float = 12.56637
    num_rays: int = 20
    min_range: float = 6.0
    max_range: float = 1.0


@dataclass
class RobotConfig:
    base_radius: float = 0.033
    vel_linear: List[float] = field(default_factory=lambda: [0.0, 1.0])
    vel_angular: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    wheel_distance: float = 0.16
    weight: float = 1.0
    threshold: float = 0.01


@dataclass
class NetworkConfig:
    hidden_size: List[int] = field(default_factory=lambda: [800, 600])


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


@dataclass
class TrainerConfig:
    batch_size: int = 64
    lr: float = 0.0001
    seed: int = 1
    num_envs: int = 1
    device: str = "mps"
    learn_step: int = 10
    target_score: float = 200.0
    max_steps: int = 1000000
    evaluation_steps: int = 10000
    evaluation_loop: int = 1
    learning_delay: int = 0
    n_step_memory: int = 1
    checkpoint: int = 100
    checkpoint_path: str = "checkpoints"
    overwrite_checkpoints: bool = False
    use_wandb: bool = False
    wandb_api_key: str = ""
    accelerator: bool = False


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


class RandomizationDomainConfig:
    weight: List[float] = field(default_factory=lambda: [2, 500])
    base_radius: List[float] = field(default_factory=lambda: [0.15, 0.75])
    wheel_distance: List[float] = field(default_factory=lambda: [0.1, 1.5])
    threshold: List[float] = field(default_factory=lambda: [0.001, 0.1])
    fov: List[float] = field(default_factory=lambda: [3.14159, 12.56637])
    num_rays: List[int] = field(default_factory=lambda: [5, 60])
    range: List[float] = field(default_factory=lambda: [0.1, 40])
    timestep: List[int] = field(default_factory=lambda: [100, 10000])
    grid_dimension: List[int] = field(default_factory=lambda: [3, 500])
    friction: List[float] = field(default_factory=lambda: [0.1, 1.0])
    porcentage_obstacles: List[float] = field(default_factory=lambda: [0.01, 0.5])
