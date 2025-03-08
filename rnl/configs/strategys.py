from abc import ABC, abstractmethod
from typing import Any, Dict, List


# Base configuration class
class BaseConfig(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


# Configuration for modes with description
class ModeConfig(BaseConfig):
    def __init__(self, mode: str, description: str):
        if not mode:
            raise ValueError("mode is required")
        if not description:
            raise ValueError("description is required")
        self.mode = mode
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "description": self.description}


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
    def __init__(self, reward: ChoiceConfig, mode: ChoiceConfig, action: ChoiceConfig):
        if not reward or not mode or not action:
            raise ValueError("reward, mode and action are required")
        self.reward = reward
        self.mode = mode
        self.action = action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": {
                "reward": self.reward.to_dict(),
                "mode": self.mode.to_dict(),
                "action": self.action.to_dict(),
            }
        }


# Single function to build and return the final JSON configuration
def get_strategy_dict() -> dict:
    modes: List[ModeConfig] = [
        ModeConfig(
            "easy-00",
            "Objetivo entre 2 posições em mapa gerado 2x2. Robô inicia aleatório (pos e ângulo 0°-360°).",
        ),
        ModeConfig(
            "easy-01",
            "Objetivo entre 4 posições em mapa gerado 2x2 fixo. Robô inicia aleatório (pos e ângulo 0°-360°).",
        ),
        ModeConfig(
            "easy-02",
            "Objetivo livre no mapa gerado 2x2. Robô inicia aleatório (pos e ângulo 0°-360°).",
        ),
        ModeConfig(
            "easy-03",
            "Objetivo livre em mapa gerado 5x5. Robô inicia aleatório (pos e ângulo 0°-360°).",
        ),
        ModeConfig(
            "easy-04",
            "Objetivo livre em mapa gerado 10x10 com obstáculos. Robô inicia aleatório (pos e ângulo 0°-360°).",
        ),
        ModeConfig(
            "easy-05",
            "Objetivo livre em mapas gerados de 2x2 até 10x10 com obstáculos. Robô inicia aleatório.",
        ),
        ModeConfig(
            "medium-00",
            "Objetivo aleatório em 1/8 do mapa real. Robô inicia aleatório (pos e ângulo).",
        ),
        ModeConfig(
            "medium-01",
            "Objetivo aleatório em 1/6 do mapa real. Robô inicia aleatório.",
        ),
        ModeConfig(
            "medium-02",
            "Objetivo aleatório em 1/4 do mapa real. Robô inicia aleatório.",
        ),
        ModeConfig(
            "medium-03",
            "Objetivo aleatório em 1/2 do mapa real. Robô inicia aleatório.",
        ),
        ModeConfig(
            "medium-04",
            "Objetivo livre em partes randômicas do mapa real. Robô inicia aleatório.",
        ),
        ModeConfig(
            "medium-05", "Objetivo livre no mapa real completo. Robô inicia aleatório."
        ),
        ModeConfig(
            "hard",
            "Mapa real gerado e objetivo totalmente randômicos. Robô inicia aleatório.",
        ),
    ]

    actions: List[ActionConfig] = [
        ActionConfig(
            "UltraSlowActions",
            "Ações ultra-lentas para máxima precisão e controle em ambientes críticos",
        ),  # 3% da velocidade normal
        ActionConfig(
            "UltraFastActions",
            "Ações ultra-rápidas para respostas imediatas em situações de alta urgência, maximizando a performance",
        ),  # 70% da velocidade normal
        ActionConfig(
            "BalancedActions",
            "Ações equilibradas que combinam velocidade moderada com estabilidade, garantindo desempenho consistente",
        ),  # 40% da velocidade normal
        ActionConfig(
            "ReactiveActions",
            "Ações reativas que se adaptam instantaneamente a mudanças do ambiente, otimizando a precisão durante as manobras",
        ),  # 35% da velocidade normal
        ActionConfig(
            "SurgeActions",
            "Ações em surto para acelerações intensas em momentos críticos, impulsionando a resposta dinâmica",
        ),  # 80% da velocidade normal
        ActionConfig(
            "SteadyActions",
            "Ações constantes que mantêm uma velocidade estável para trajetórias suaves e seguras",
        ),  # 25% da velocidade normal
        ActionConfig(
            "DriftActions",
            "Ações de drift que proporcionam curvas dinâmicas e controladas, combinando agilidade e segurança em manobras acentuadas",
        ),  # 30% da velocidade normal
        ActionConfig(
            "PrecisionCurveActions",
            "Ações de curva precisas que ajustam a velocidade ideal para cada ângulo, garantindo máxima aderência e controle",
        ),  # 15% da velocidade normal
    ]

    rewards: List[RewardConfig] = [
        RewardConfig("time", {"scale_time": 0.01}, "Recompensa negativa a cada step"),
        RewardConfig(
            "distance",
            {"scale_distance": 0.1},
            "Recompensa com base na diferenca entre a distancia inicial (reset) e a posicao atual em relacao ao objetivo",
        ),
        RewardConfig(
            "orientation",
            {"scale_orientation": 0.003},
            "Recompensa positiva do robo em relacao ao objetivo",
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
    mode_choice = ChoiceConfig(modes, required=True)
    action_choice = ChoiceConfig(actions, required=True)

    strategy = StrategyConfig(
        reward=reward_choice, mode=mode_choice, action=action_choice
    )
    return strategy.to_dict()
