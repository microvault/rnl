import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List


# Classe base para configurações
class BaseConfig(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


# Configuração para modos com descrição
class ModeConfig(BaseConfig):
    def __init__(self, mode: str, description: str):
        if not mode:
            raise ValueError("mode é obrigatório")
        if not description:
            raise ValueError("description é obrigatório")
        self.mode = mode
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "description": self.description, "required": True}


# Configuração para ações com descrição
class ActionConfig(BaseConfig):
    def __init__(self, action_type: str, parameters: Dict[str, Any], description: str):
        if not action_type:
            raise ValueError("action_type é obrigatório")
        if parameters is None:
            raise ValueError("parameters é obrigatório")
        if not description:
            raise ValueError("description é obrigatório")
        self.action_type = action_type
        self.parameters = parameters
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
            "description": self.description,
            "required": True,
        }


# Configuração para recompensas com descrição
class RewardConfig(BaseConfig):
    def __init__(self, reward_type: str, parameters: Dict[str, Any], description: str):
        if not reward_type:
            raise ValueError("reward_type é obrigatório")
        if parameters is None:
            raise ValueError("parameters é obrigatório")
        if not description:
            raise ValueError("description é obrigatório")
        self.reward_type = reward_type
        self.parameters = parameters
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_type": self.reward_type,
            "parameters": self.parameters,
            "description": self.description,
            "required": True,
        }


# Listas separadas para modos, ações e recompensas
modes: List[ModeConfig] = [
    ModeConfig("easy-00", "Modo fácil com configurações básicas"),
    ModeConfig("easy-01", "Modo fácil com opções variadas de alvo"),
    ModeConfig("medium", "Modo intermediário com desafios mais complexos"),
]

actions: List[ActionConfig] = [
    ActionConfig("SlowActions", {"speed": "slow"}, "Ações lentas para maior precisão"),
    ActionConfig(
        "FastActions", {"speed": "fast"}, "Ações rápidas para respostas imediatas"
    ),
]

rewards: List[RewardConfig] = [
    RewardConfig("time", {"scale_time": 0.01}, "Recompensa baseada no tempo gasto"),
    RewardConfig(
        "distance",
        {"scale_distance": 0.1},
        "Recompensa baseada na distância percorrida",
    ),
    RewardConfig(
        "all",
        {"scale_distance": 0.1, "scale_orientation": 0.003},
        "Recompensa combinada de tempo, distância e orientação",
    ),
]

# Gera o JSON agrupado
config = {
    "modes": [mode.to_dict() for mode in modes],
    "actions": [action.to_dict() for action in actions],
    "rewards": [reward.to_dict() for reward in rewards],
}

json_config = json.dumps(config, indent=4)
print(json_config)
