from abc import ABC, abstractmethod
from typing import Any, Dict


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
    def __init__(self, parameters: Dict[str, Any]):
        if parameters is None:
            raise ValueError("parameters are required")
        self.parameters = parameters

    def to_dict(self) -> Dict[str, Any]:
        # Converte os parâmetros para uma lista com "key" e "default"
        param_list = [
            {"key": key, "default": default_value}
            for key, default_value in self.parameters.items()
        ]
        return {
            "parameters": param_list,
        }
