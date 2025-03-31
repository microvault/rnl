from abc import ABC, abstractmethod
from typing import Any, Dict, List


# Base configuration class
class BaseConfig(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class DomainConfig(BaseConfig):
    def __init__(
        self,
        obstacle_percentage: int,
        obstacle_percentage_description: str,
        map_size: float,
        map_size_description: str,
    ):
        if not isinstance(obstacle_percentage, int):
            raise ValueError("obstacle_percentage deve ser um int")
        if obstacle_percentage < 0 or obstacle_percentage > 100:
            raise ValueError("obstacle_percentage deve estar entre 0 e 100")
        if not obstacle_percentage_description:
            raise ValueError("obstacle_percentage_description é obrigatório")
        if not isinstance(map_size, float):
            raise ValueError("map_size deve ser um float")
        if map_size <= 0:
            raise ValueError("map_size deve ser maior que zero")
        if not map_size_description:
            raise ValueError("map_size_description é obrigatório")

        self.obstacle_percentage = obstacle_percentage
        self.obstacle_percentage_description = obstacle_percentage_description
        self.map_size = map_size
        self.map_size_description = map_size_description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obstacle_percentage": {
                "value": self.obstacle_percentage,
                "description": self.obstacle_percentage_description,
            },
            "map_size": {
                "value": self.map_size,
                "description": self.map_size_description,
            },
        }


# Configuration for actions with description
class ActionConfig(BaseConfig):
    def __init__(self, action_type: str, description: str):
        if not action_type:
            raise ValueError("action_type is required")
        if not description:
            raise ValueError("description is required")
        self.action_type = action_type
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {"action_type": self.action_type, "description": self.description}


# Configuration for rewards with description
class RewardConfig(BaseConfig):
    def __init__(self, reward_type: str, parameters: Dict[str, Any], description: str):
        if not reward_type:
            raise ValueError("reward_type is required")
        if parameters is None:
            raise ValueError("parameters are required")
        if not description:
            raise ValueError("description is required")
        self.reward_type = reward_type
        self.parameters = parameters
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        # Converte os parâmetros para uma lista com "key" e "default"
        param_list = [
            {"key": key, "default": default_value}
            for key, default_value in self.parameters.items()
        ]
        return {
            "reward_type": self.reward_type,
            "parameters": param_list,
            "description": self.description,
        }


# Class to group the options, with a group-level required flag
class ChoiceConfig:
    def __init__(self, options: List[BaseConfig], required: bool = True):
        self.options = options
        self.required = required

    def to_dict(self) -> Dict[str, Any]:
        d = {"options": [item.to_dict() for item in self.options]}
        if self.required:
            d["required"] = True
        return d


# Main Strategy class, where reward, mode and action are required
class StrategyConfig:
    def __init__(self, reward: ChoiceConfig, mode: ChoiceConfig):
        if not reward or not mode:
            raise ValueError("reward, mode and action are required")
        self.reward = reward
        self.mode = mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": {
                "reward": self.reward.to_dict(),
                "mode": self.mode.to_dict(),
            }
        }


# Single function to build and return the final JSON configuration
def get_strategy_dict() -> dict:
    domains: List[DomainConfig] = [
        DomainConfig(0, "Sem obstáculos", 1.0, "Mapa 1x1"),
        DomainConfig(25, "Poucos obstáculos", 2.0, "Mapa 2x2"),
        DomainConfig(50, "Obstáculos moderados", 3.0, "Mapa 3x3"),
        DomainConfig(75, "Muitos obstáculos", 4.0, "Mapa 4x4"),
        DomainConfig(100, "Máximo de obstáculos", 5.0, "Mapa 5x5"),
    ]

    rewards: List[RewardConfig] = [
        RewardConfig("time", {"scale_time": 0.01}, "Recompensa negativa a cada step"),
        RewardConfig(
            "distance",
            {"scale_distance": 0.1},
            "Recompensa dada quando mais perto o robô chega ao objetivo",
        ),
        RewardConfig(
            "orientation",
            {"scale_orientation": 0.003},
            "Recompensa positiva se o robo estiver em direcao ao objetivo",
        ),
        RewardConfig(
            "any",
            {},
            "Somente a recompensa negativa -1 quando colide e +1 quando chega ao objetivo",
        ),
        RewardConfig(
            "distance_orientation",
            {"scale_distance": 0.1, "scale_orientation": 0.003},
            "Combinacao de distancia e orientacao",
        ),
        RewardConfig(
            "distance_time",
            {"scale_distance": 0.1, "scale_time": 0.01},
            "Combinacao de distancia e tempo",
        ),
        RewardConfig(
            "orientation_time",
            {"scale_orientation": 0.003, "scale_time": 0.01},
            "Combinacao de orientacao e tempo",
        ),
        RewardConfig(
            "distance_orientation_time",
            {"scale_distance": 0.1, "scale_orientation": 0.003, "scale_time": 0.01},
            "Reward based on distance, orientation and time",
        ),
        RewardConfig(
            "distance_obstacle",
            {"scale_distance": 0.1, "scale_obstacle": 0.001},
            "Reward based on distance and obstacles",
        ),
        RewardConfig(
            "orientation_obstacle",
            {"scale_orientation": 0.003, "scale_obstacle": 0.001},
            "Reward based on orientation and obstacles",
        ),
        RewardConfig(
            "all",
            {
                "scale_distance": 0.1,
                "scale_orientation": 0.003,
                "scale_time": 0.01,
                "scale_obstacle": 0.001,
            },
            "Reward based on all factors",
        ),
    ]

    reward_choice = ChoiceConfig(rewards, required=True)
    domain_choice = ChoiceConfig(domains, required=True)

    strategy = StrategyConfig(reward=reward_choice, mode=domain_choice)

    return strategy.to_dict()
