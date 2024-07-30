from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import inf

from microvault.components.replaybuffer import ReplayBuffer
from microvault.models.model import ModelActor, ModelCritic


class Agent:
    """Treinamento do agente com o Ambiente."""

    def __init__(
        self,
        modelActor: ModelActor,
        modelCritic: ModelCritic,
        state_size: int = 14,
        action_size: int = 2,
        max_action: float = 1.0,
        min_action: float = -1.0,
        update_every_step: int = 2,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        weight_decay: float = 0.0,
        noise: float = 0.2,
        noise_clip: float = 0.5,
        device: str = "cpu",
        pretrained: bool = False,
        nstep: int = 1,
    ):
        """Inicializar o agente.

        Parâmetros:
        ======
            state_size (int): Tamanho do espaço de estado
            action_size (int): Tamanho do espaço de ação
            max_action (ndarray): Valor maximo valido para cada vector de ação
            min_action (ndarray): Valor minimo valido para cada vector de ação
            noise (float): Valor de ruído gerado na política
            noise_clip (float): Cortar ruído aleatório neste intervalo
        """

        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.min_action = min_action
        self.gamma = gamma
        self.tau = tau
        self.update_every_step = update_every_step
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.noise = noise
        self.noise_clip = noise_clip
        self.device = device
        self.pretrained = pretrained
        self.nstep = nstep

        # Transfência de aprendizado
        if self.pretrained:

            # Rede "Actor" (com/ Rede Alvo)
            self.actor = modelActor.to(self.device)
            self.actor.load_state_dict(torch.load("/content/checkpoint_actor.pth"))
            # Definir o modelo alvo, porém, sem a necessidade de calcular gradientes
            self.actor_target = modelActor.to(
                self.device
            )  # .eval().requires_grad_(False)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

            # Rede "Actor" (com/ Rede Alvo)
            self.critic = modelCritic.to(self.device)
            self.critic.load_state_dict(
                torch.load("/content/checkpoint_critic.pth", map_location=self.device)
            )
            self.critic_target = modelCritic.to(
                self.device
            )  # .eval().requires_grad_(False)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(),
                lr=self.lr_critic,
                weight_decay=self.weight_decay,
            )

        else:
            # Rede "Actor" (com/ Rede Alvo)
            self.actor = modelActor.to(self.device)
            self.actor_target = modelActor.to(
                self.device
            )  # .eval().requires_grad_(False)

            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=self.lr_critic
            )

            # Rede "Actor" (com/ Rede Alvo)
            self.critic = modelCritic.to(self.device)
            self.critic_target = modelCritic.to(
                self.device
            )  # .eval().requires_grad_(False)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(),
                lr=self.lr_critic,
                weight_decay=self.weight_decay,
            )

        self.clip_grad = torch.nn.utils.clip_grad_norm_

        # TODO: Inicializar o modelo RND

    def predict(self, states: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
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
        self.actor.eval()

        # Desativar o cálculo de gradientes
        with torch.no_grad():
            # Passar o estado para o modelo e retorna a ação em np.ndarray
            actions_without_noise = self.actor(state).cpu().data.numpy()

        # Deixar a ação dentro dos limites
        action = actions_without_noise.clip(self.min_action, self.max_action)

        assert (
            action.shape[0] == self.action_size
        ), "O tamanho da ação é diferente do tamanho definido em PREDICT."

        assert isinstance(
            action[0], np.float32
        ), "Action is not of type (np.float32) in PREDICT -> action type: {}.".format(
            type(action)
        )
        return action

    def learn(
        self, memory: ReplayBuffer, n_iteration: int
    ) -> Tuple[float, float, float, float, float, float]:
        """Atualize parâmetros de política e valor usando determinado lote de tuplas de experiência.

        Parâmetros
        ======
            n_iteraion (int): O número de iterações para treinar a rede
        """

        self.actor.train()
        self.critic.train()

        average_Q = 0
        max_Q = -inf
        average_critic_loss = 0
        average_actor_loss = 0
        intrinsic_reward = 0
        errors = 0
        actor_loss = torch.tensor(0)

        # Percorrer o número de iterações
        for i in range(n_iteration):  # loop:
            # ---------------------------- Amostragem de Experiência ---------------------------- #
            # Obtero o índice da amostra, a amostra (s, a, r, s', d) e os pesos de importância
            state, action, reward, next_state, done = memory.sample()

            # Converter a ação para tensor
            action_ = action.cpu().numpy()

            # TODO: RND

            # ---------------------------- Atualizar crítico ---------------------------- #
            # Obtenha ações de próximo estado previstas e valores Q de modelos de destino

            # Desativar o cálculo de gradientes
            with torch.no_grad():

                # Gerar um ruído aleatório para a ação
                noise = (
                    torch.FloatTensor(action_)
                    .data.normal_(0, self.noise)
                    .to(self.device)
                )
                # Deixar a ação dentro dos limites
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                # Obter a próxima ação do ator de destino com ruído
                actions_next = (self.actor_target(next_state) + noise).clamp(
                    float(self.min_action),
                    float(self.max_action),
                )

                # Obter 2 Q-valor de destino para o próximo estado
                Q1_targets_next, Q2_targets_next = self.critic_target(
                    next_state, actions_next
                )

                # Obter o menor Q-valor de destino entre os 2 Q-valores
                Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)

                # Média dos valores Q
                average_Q += torch.mean(Q_targets_next)
                # Máximo dos valores Q
                max_Q = max(max_Q, torch.max(Q_targets_next))

                # Calcular metas Q para estados atuais (y_i)
                # recompensa + (gamma^nstep * Q-valor de destino * (1 - feito))
                Q_targets = (
                    reward
                    + (self.gamma**self.nstep * Q_targets_next * (1 - done)).detach()
                )

            # Passar o estado atual e ação para o modelo crítico
            Q1_expected, Q2_expected = self.critic(state, action)
            # Valor Q esperado minimo
            Q_expected = torch.min(Q1_expected, Q2_expected)
            # Error absoluto entre os valores Q esperados e os valores Q alvos em np.adarray
            errors = np.abs((Q_expected - Q_targets).detach().cpu().numpy())

            # TODO: random intrinsic reward 0 a 10
            intrinsic_reward = torch.rand(1) * 10
            # Calcular a perda do crítico
            critic_loss = F.mse_loss(Q1_expected, Q_targets) + F.mse_loss(
                Q2_expected, Q_targets
            )

            # Minimize a perda
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if i % self.update_every_step == 0:
                # ---------------------------- atualizar Ator ---------------------------- #
                # Calcular perda de ator
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Minimize a perda
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ----------------------- Atualizar redes de destino ----------------------- #
                # Passar todos os pesos da rede de destino para a rede local usando a atualização suave
                self.soft_update(self.critic, self.critic_target, self.tau)
                self.soft_update(self.actor, self.actor_target, self.tau)

                average_actor_loss += actor_loss

            average_critic_loss += critic_loss

        loss_critic = average_critic_loss / n_iteration
        loss_actor = average_actor_loss / n_iteration
        average_intrinsic_reward = intrinsic_reward / n_iteration
        average_policy = average_Q / n_iteration
        average_errors = np.mean(errors / n_iteration)
        max_policy = max_Q

        # delete variables
        del (
            average_Q,
            max_Q,
            average_critic_loss,
            average_actor_loss,
            intrinsic_reward,
            errors,
        )

        return (
            float(loss_critic),
            float(loss_actor),
            float(average_policy),
            float(max_policy),
            float(average_intrinsic_reward),
            float(average_errors),
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

    def save(self, filename: str = "checkpoint", version: str = "latest") -> None:
        """Save the model"""
        torch.save(self.critic.state_dict(), filename + "_critic_" + version + ".pth")
        torch.save(
            self.critic_optimizer.state_dict(),
            filename + "_critic_optimizer_" + version + ".pth",
        )

        torch.save(self.actor.state_dict(), filename + "_actor_" + version + ".pth")
        torch.save(
            self.actor_optimizer.state_dict(),
            filename + "_actor_optimizer_" + version + ".pth",
        )
