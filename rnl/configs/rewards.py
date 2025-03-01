from dataclasses import dataclass


@dataclass
class RewardConfig:
    reward_type: str  # "time", "distance", etc.
    scale: float = 1.0
    # Pode acrescentar parâmetros extras
