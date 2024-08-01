import torch
import torch.nn as nn
import torch.nn.functional as F


class QModel(nn.Module):
    """Modelo de Ator (Política)"""

    def __init__(
        self,
        state_size: int = 13,
        action_size: int = 4,
        fc1_units: int = 128,
        fc2_units: int = 32,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """Inicializar os parâmetros e construir o modelo.
        parâmetros
        ======
                state_size: tamanho do espaço de estado.
                action_size: tamanho do espaço de ação.
                semente (int): semente aleatória
                fc1_units (int): Número de nós na primeira camada oculta
                fc2_units (int): Número de nós na segunda camada oculta
        """
        super().__init__()
        self.state_size = state_size
        self.batch_size = batch_size
        self.device = device

        self.fc1 = nn.Linear(state_size, fc1_units)  # camada de entrda com 128 nos
        self.fc2 = nn.Linear(
            fc1_units, fc2_units
        )  # camada oculta com 128 de entrada e 32 de saida
        self.fc3 = nn.Linear(
            fc2_units, action_size
        )  # camada de saida com acoes possiveis

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Construir uma rede que mapeia estado -> valores de ação."""

        assert isinstance(
            state, torch.Tensor
        ), "State is not of type torch.Tensor in Network."
        assert (
            state.dtype == torch.float32
        ), "Tensor elements are not of type torch.float32 in Network."
        assert (
            state.shape[0] <= self.state_size or state.shape[0] >= self.batch_size
        ), f"The tensor shape is not torch.Size([24]) in Network., {state.shape[0]}"
        assert (
            str(state.device.type) == self.device
        ), "The state must be on the same device in Network."
        x = F.relu(self.fc1(state))  # funcao de ativação relu
        x = F.relu(self.fc2(x))  # funcao de ativação relu
        return F.softmax(self.fc3(x), dim=-1)  # retorna a camada de saida
