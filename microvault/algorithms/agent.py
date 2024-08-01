import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import inf

from microvault.components.replaybuffer import ReplayBuffer
from microvault.models.model import QModel


class Agent:
    """Treinamento do agente com o Ambiente."""

    def __init__(
        self,
        model: QModel,
        state_size: int = 13,
        action_size: int = 4,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr_model: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        pretrained: bool = False,
    ):
        """Inicializar o agente.

        Parâmetros:
        ======
            state_size (int): Tamanho do espaço de estado
            action_size (int): Tamanho do espaço de ação
        """

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.lr_model = lr_model
        self.weight_decay = weight_decay
        self.device = device
        self.pretrained = pretrained

        # Transfência de aprendizado
        if self.pretrained:
            # Q-Network
            self.qnetwork_local = model.to(self.device)

            self.qnetwork_local.load_state_dict(
                torch.load(
                    "/Users/nicolasalan/microvault/microvault/checkpoints/checkpoint.pth"
                )
            )

            self.qnetwork_target = model.to(self.device)
            self.optimizer = optim.Adam(
                self.qnetwork_local.parameters(), lr=self.lr_model
            )

        else:
            # Q-Network
            self.qnetwork_local = model.to(self.device)
            self.qnetwork_target = model.to(self.device)
            self.optimizer = optim.Adam(
                self.qnetwork_local.parameters(), lr=self.lr_model
            )

    def predict(
        self, states: np.ndarray[np.float32], eps: float = 0.0
    ) -> np.ndarray[np.float32]:
        """Retorna ações para determinado estado de acordo com a política atual."""

        assert isinstance(
            states, np.ndarray
        ), "Os estados não são de estrutura de dados (np.ndarray) em PREDICT -> estados: {}.".format(
            type(states)
        )
        assert isinstance(
            states[0], np.float32
        ), "Estado não é do tipo (np.float32) em PREDICT -> states type: {}.".format(
            type(states)
        )
        assert (
            states.shape[0] == self.state_size
        ), f"O Tamanho dos estados não é {self.state_size} em PREDICT -> states size: {states.shape[0]}.".format(
            states.shape[0]
        )
        assert (
            states.ndim == 1
        ), "O ndim dos estados não é (1) em PREDICT -> estados ndim: {}.".format(
            states.ndim
        )

        # Converter estados para tensor
        state = torch.from_numpy(states).float().to(self.device)

        # Desativar o cálculo de gradientes
        self.qnetwork_local.eval()

        # Desativar o cálculo de gradientes
        with torch.no_grad():
            # Passar o estado para o modelo e retorna a ação em np.ndarray
            action_values = self.qnetwork_local(state).cpu().data.numpy()

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:  # se o número aleatório for maior que o epsilon
            return np.argmax(action_values)  # retornar a ação com maior valor
        else:
            return random.choice(np.arange(self.action_size))

    def learn(
        self, memory: ReplayBuffer, n_iteration: int
    ) -> Tuple[float, float, float]:
        """Atualize parâmetros de política e valor usando determinado lote de tuplas de experiência.

        Parâmetros
        ======
            n_iteraion (int): O número de iterações para treinar a rede
        """

        average_Q = 0
        max_Q = -inf
        average_model_loss = 0
        loss = torch.tensor(0)

        # Percorrer o número de iterações
        for i in range(n_iteration):  # loop:
            # ---------------------------- Amostragem de Experiência ---------------------------- #
            # Obtero o índice da amostra, a amostra (s, a, r, s', d) e os pesos de importância
            state, action, reward, next_state, done = memory.sample()

            Q_targets_next = (
                self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
            )
            average_Q += torch.mean(Q_targets_next)
            max_Q = max(max_Q, torch.max(Q_targets_next))
            # Calcula Q alvos para os estados atuais
            # y = r + γ * maxQhat
            Q_targets = reward + (self.gamma * Q_targets_next * (1 - done))

            # Obtenha os valores Q esperados do modelo local
            # Q(\phi(s_t), a_j; \theta)
            Q_expected = self.qnetwork_local(state).gather(1, action)

            # Perda de cálculo
            # execute gradient descent step at (y - Q)**2
            loss = F.mse_loss(Q_expected, Q_targets)
            # Minimizar a perda
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

            average_model_loss += loss

        loss_all = average_model_loss / n_iteration
        average_policy = average_Q / n_iteration
        max_policy = max_Q

        # delete variables
        del (
            average_Q,
            max_Q,
        )

        return (
            float(loss_all),
            float(average_policy),
            float(max_policy),
        )

    @staticmethod
    def soft_update(
        local_model: nn.Module, target_model: nn.Module, tau: float = 1e-3
    ) -> None:
        """Parâmetros do modelo de atualização suave.
        θ_alvo = τ * θ_local + (1 - τ) * θ_alvo
        Parâmetros
        ======
            local_model: modelo PyTorch (os pesos serão copiados)
            target_model: modelo PyTorch (os pesos serão copiados)
            tau (float): parâmetro de interpolação
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save(
        self, filename: str = "checkpoints/checkpoint", version: str = "latest"
    ) -> None:
        """Save the model"""
        if not os.path.dirname(filename):
            filename = os.path.join("checkpoints", filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        torch.save(
            self.qnetwork_local.state_dict(), filename + "_model_" + version + ".pth"
        )
        torch.save(
            self.optimizer.state_dict(),
            filename + "_model_optimizer_" + version + ".pth",
        )
