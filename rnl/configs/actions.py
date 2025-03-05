from abc import ABC, abstractmethod
from typing import Tuple


def interpolate(min_val: float, max_val: float, factor: float) -> float:
    """
    Retorna um valor interpolado entre min_val e max_val, usando o fator (0 a 1).
    """
    return min_val + factor * (max_val - min_val)


class ActionsConfig(ABC):
    """
    Interface que toda classe de ações precisa implementar.
    """

    @abstractmethod
    def get_speeds(
        self,
        action_id: int,
        min_value_lr: float,
        max_value_lr: float,
        min_value_vr: float,
        max_value_vr: float,
    ) -> Tuple[float, float]:
        """
        Retorna (velocidade_linear, velocidade_angular) para o ID de ação dado (0, 1 ou 2).
        """
        pass


class UltraSlowActions(ActionsConfig):
    """
    Ações ultra-lentas para máxima precisão e controle em ambientes críticos (10% da velocidade normal).
    """

    MULTIPLIER = 0.10

    def get_speeds(
        self,
        action_id: int,
        min_value_lr: float,
        max_value_lr: float,
        min_value_vr: float,
        max_value_vr: float,
    ) -> Tuple[float, float]:
        if action_id == 0:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            return (speed_lr, 0.0)
        elif action_id == 1:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = -interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 8
            return (speed_lr, speed_vr)
        elif action_id == 2:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 8
            return (speed_lr, speed_vr)

        return (0.0, 0.0)


class UltraFastActions(ActionsConfig):
    """
    Ações ultra-rápidas para respostas imediatas em situações de alta urgência, maximizando a performance (70% da velocidade normal).
    """

    MULTIPLIER = 0.70

    def get_speeds(
        self,
        action_id: int,
        min_value_lr: float,
        max_value_lr: float,
        min_value_vr: float,
        max_value_vr: float,
    ) -> Tuple[float, float]:
        if action_id == 0:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            return (speed_lr, 0.0)
        elif action_id == 1:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = -interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 4
            return (speed_lr, speed_vr)
        elif action_id == 2:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 4
            return (speed_lr, speed_vr)
        return (0.0, 0.0)


class BalancedActions(ActionsConfig):
    """
    Ações equilibradas que combinam velocidade moderada com estabilidade, garantindo desempenho consistente (40% da velocidade normal).
    """

    MULTIPLIER = 0.40

    def get_speeds(
        self,
        action_id: int,
        min_value_lr: float,
        max_value_lr: float,
        min_value_vr: float,
        max_value_vr: float,
    ) -> Tuple[float, float]:
        if action_id == 0:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            return (speed_lr, 0.0)
        elif action_id == 1:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = -interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 4
            return (speed_lr, speed_vr)
        elif action_id == 2:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 4
            return (speed_lr, speed_vr)

        return (0.0, 0.0)


class ReactiveActions(ActionsConfig):
    """
    Ações reativas que se adaptam instantaneamente a mudanças do ambiente, otimizando a precisão durante as manobras (35% da velocidade normal).
    """

    MULTIPLIER = 0.35

    def get_speeds(
        self,
        action_id: int,
        min_value_lr: float,
        max_value_lr: float,
        min_value_vr: float,
        max_value_vr: float,
    ) -> Tuple[float, float]:
        if action_id == 0:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            return (speed_lr, 0.0)
        elif action_id == 1:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = -interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 2
            return (speed_lr, speed_vr)
        elif action_id == 2:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 2
            return (speed_lr, speed_vr)
        return (0.0, 0.0)


class SurgeActions(ActionsConfig):
    """
    Ações em surto para acelerações intensas em momentos críticos, impulsionando a resposta dinâmica (80% da velocidade normal).
    """

    MULTIPLIER = 0.80

    def get_speeds(
        self,
        action_id: int,
        min_value_lr: float,
        max_value_lr: float,
        min_value_vr: float,
        max_value_vr: float,
    ) -> Tuple[float, float]:
        if action_id == 0:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            return (speed_lr, 0.0)
        elif action_id == 1:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = -interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 4
            return (speed_lr, speed_vr)
        elif action_id == 2:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 4
            return (speed_lr, speed_vr)
        return (0.0, 0.0)


class SteadyActions(ActionsConfig):
    """
    Ações constantes que mantêm uma velocidade estável para trajetórias suaves e seguras (25% da velocidade normal).
    """

    MULTIPLIER = 0.25

    def get_speeds(
        self,
        action_id: int,
        min_value_lr: float,
        max_value_lr: float,
        min_value_vr: float,
        max_value_vr: float,
    ) -> Tuple[float, float]:
        if action_id == 0:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            return (speed_lr, 0.0)
        elif action_id == 1:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = -interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 4
            return (speed_lr, speed_vr)
        elif action_id == 2:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 4
            return (speed_lr, speed_vr)
        return (0.0, 0.0)


class DriftActions(ActionsConfig):
    """
    Ações de drift que proporcionam curvas dinâmicas e controladas, combinando agilidade e segurança em manobras acentuadas (30% da velocidade normal).
    """

    MULTIPLIER = 0.30

    def get_speeds(
        self,
        action_id: int,
        min_value_lr: float,
        max_value_lr: float,
        min_value_vr: float,
        max_value_vr: float,
    ) -> Tuple[float, float]:
        if action_id == 0:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            return (speed_lr, 0.0)
        elif action_id == 1:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = -interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 6
            return (speed_lr, speed_vr)
        elif action_id == 2:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 6
            return (speed_lr, speed_vr)
        return (0.0, 0.0)


class PrecisionCurveActions(ActionsConfig):
    """
    Ações de curva precisas que ajustam a velocidade ideal para cada ângulo, garantindo máxima aderência e controle (15% da velocidade normal).
    """

    MULTIPLIER = 0.15

    def get_speeds(
        self,
        action_id: int,
        min_value_lr: float,
        max_value_lr: float,
        min_value_vr: float,
        max_value_vr: float,
    ) -> Tuple[float, float]:
        if action_id == 0:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            return (speed_lr, 0.0)
        elif action_id == 1:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = -interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 4
            return (speed_lr, speed_vr)
        elif action_id == 2:
            speed_lr = interpolate(min_value_lr, max_value_lr, self.MULTIPLIER)
            speed_vr = interpolate(min_value_vr, max_value_vr, self.MULTIPLIER) / 4
            return (speed_lr, speed_vr)
        return (0.0, 0.0)


def get_actions_class(class_name: str):
    """
    Retorna a classe correspondente ao nome fornecido.

    Parâmetros:
      - class_name: Nome da classe em string.

    Retorna:
      - A classe correspondente se encontrada.

    Exceções:
      - ValueError: Se a classe não for encontrada.
    """
    classes = {
        "UltraSlowActions": UltraSlowActions,
        "UltraFastActions": UltraFastActions,
        "BalancedActions": BalancedActions,
        "ReactiveActions": ReactiveActions,
        "SurgeActions": SurgeActions,
        "SteadyActions": SteadyActions,
        "DriftActions": DriftActions,
        "PrecisionCurveActions": PrecisionCurveActions,
    }
    if class_name in classes:
        return classes[class_name]
    else:
        raise ValueError(f"Classe '{class_name}' não encontrada.")
