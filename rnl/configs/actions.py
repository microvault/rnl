from abc import ABC, abstractmethod
from typing import Tuple

# ========================================= #
# 1. Classe base abstrata para as ações     #
# ========================================= #
class ActionsConfig(ABC):
    """
    Define a interface (métodos) que toda classe de ações precisa ter.
    """

    @abstractmethod
    def get_speeds(self, action_id: int) -> Tuple[float, float]:
        """
        Retorna (vl, vr) para o ID de ação dado (0, 1 ou 2).
        """
        pass


# ============================================= #
# 2. Diferentes implementações de "BaseActions" #
# ============================================= #

class SlowActions(ActionsConfig):
    """
    3 ações:
      - Ação 0: Forward lento
      - Ação 1: Gira lento para a direita
      - Ação 2: Gira lento para a esquerda
    """
    def get_speeds(self, action_id: int) -> Tuple[float, float]:
        if action_id == 0:
            return (0.1, 0.0)       # forward lento
        elif action_id == 1:
            return (0.05, -0.15)    # vira devagar direita
        elif action_id == 2:
            return (0.05, 0.15)     # vira devagar esquerda
        # Caso default
        return (0.0, 0.0)

class FastActions(ActionsConfig):
    """
    3 ações:
      - Ação 0: Forward rápido
      - Ação 1: Gira rápido p/ direita
      - Ação 2: Gira rápido p/ esquerda
    """
    def get_speeds(self, action_id: int) -> Tuple[float, float]:
        if action_id == 0:
            return (0.3, 0.0)       # forward rápido
        elif action_id == 1:
            return (0.2, -0.5)      # vira rápido direita
        elif action_id == 2:
            return (0.2, 0.5)       # vira rápido esquerda
        return (0.0, 0.0)

class FastActions(ActionsConfig):
    """
    3 ações:
      - Ação 0: Forward rápido
      - Ação 1: Gira rápido p/ direita
      - Ação 2: Gira rápido p/ esquerda
    """
    def get_speeds(self, action_id: int) -> Tuple[float, float]:
        if action_id == 0:
            return (0.3, 0.0)       # forward rápido
        elif action_id == 1:
            return (0.2, -0.5)      # vira rápido direita
        elif action_id == 2:
            return (0.2, 0.5)       # vira rápido esquerda
        return (0.0, 0.0)
'
