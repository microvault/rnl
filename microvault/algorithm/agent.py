import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from components.replaybuffer import ReplayBuffer
from models.model import ModelActor, ModelCritic
from numpy import inf


class Agent:
    """Treinamento do agente com o Ambiente."""

    def __init__(
        self,
        replayBuffer=ReplayBuffer,
        modelActor=ModelActor,
        modelCritic=ModelCritic,
        state_size: int = 13,
        action_size: int = 2,
        max_action: float = 1.0,
        min_action: float = -1.0,
        batch_size: int = 128,
        update_every_step: int = 2,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        weight_decay: float = 0.0,
        noise: float = 0.2,
        noise_std: float = 0.1,
        noise_clip: float = 0.5,
        random_seed: int = 42,
        device: str = "cpu",
        pretrained: bool = False,
        nstep: int = 1,
        desired_distance: float = 0.7,
        scalar: float = 0.05,
        scalar_decay: float = 0.99,
    ):
        """Inicializar o agente.

        Parâmetros:
        ======
            state_size (int): Tamanho do espaço de estado
            action_size (int): Tamanho do espaço de ação
            max_action (ndarray): Valor maximo valido para cada vetor de ação
            min_action (ndarray): Valor minimo valido para cada vetor de ação
            noise (float): Valor de ruído gerado na política
            noise_std (float): Desvio padrão do ruído
            noise_clip (float): Cortar ruído aleatório neste intervalo
        """

        # modelActor = ModelActor()
        # modelCritic = ModelCritic()

        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.min_action = min_action
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every_step = update_every_step
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.noise = noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.random_seed = random_seed
        self.device = device
        self.pretrained = pretrained
        self.nstep = nstep
        self.desired_distance = desired_distance
        self.scalar = scalar
        self.scalar_decay = scalar_decay

        # Parâmetros do ruído
        self.distances = []

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

            # Rede "Actor" para ruído
            self.actor_noised = modelActor.to(self.device)

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
                self.critic.parameters(), lr=self.lr_critic
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

            self.actor_noised = modelActor.to(self.device)

            # Rede "Actor" (com/ Rede Alvo)
            self.critic = modelCritic.to(self.device)
            self.critic_target = modelCritic.to(
                self.device
            )  # .eval().requires_grad_(False)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=self.lr_critic
            )

        self.clip_grad = torch.nn.utils.clip_grad_norm_

        # Memória de replay priorizada
        # Fonte: https://arxiv.org/abs/1511.05952
        self.memory = replayBuffer

        # TODO: Inicializar o modelo RND

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Salvar experiência na memória de replay (estado, ação, recompensa, próximo estado, feito)."""

        self.memory.add(state, action, reward, next_state, done)

    def predict(self, states: np.ndarray) -> np.ndarray:
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
        self.actor_noised.eval()

        # Desativar o cálculo de gradientes
        with torch.no_grad():
            # Passar o estado para o modelo e retorna a ação em np.ndarray
            action = self.actor(state).cpu().data.numpy()

            # ---------------------------- Adicionar Ruído as Camadas do Modelo ---------------------------- #
            # Fonte: https://arxiv.org/abs/1706.01905

            # Carregar os pesos do modelo para o modelo com ruído
            self.actor_noised.load_state_dict(self.actor.state_dict().copy())
            # Adicionar ruído ao modelo
            self.actor_noised.add_parameter_noise(self.scalar)
            # Obtenha os próximos valores de ação do ator barulhento
            action_noised = self.actor_noised(state).cpu().data.numpy()
            # Mede a distância entre os valores de ação dos ator regulares e ator ruidoso
            distance = np.sqrt(np.mean(np.square(action - action_noised)))
            # Adicionar a distância ao histórico de distâncias
            self.distances.append(distance)
            # Ajuste a quantidade de ruído dada ao "actor_noised"
            if distance > self.desired_distance:
                self.scalar *= self.scalar_decay
            if distance < self.desired_distance:
                self.scalar /= self.scalar_decay
            # Definir a ação barulhenta como ação
            action = action_noised

        # Ativar o cálculo de gradientes
        self.actor.train()

        # Deixar a ação dentro dos limites
        action = action.clip(self.min_action, self.max_action)

        assert (
            action.shape[0] == self.action_size
        ), "O tamanho da ação é diferente do tamanho definido em PREDICT."

        ## assert isinstance(action[0], np.float32), "Action is not of type (np.float32) in PREDICT -> action type: {}.".format(type(action))

        return action

    def learn(
        self, n_iteraion: int, episode: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Atualize parâmetros de política e valor usando determinado lote de tuplas de experiência.

        Parâmetros
        ======
            n_iteraion (int): O número de iterações para treinar a rede
        """
        self.actor.train()
        self.critic.train()

        # Se o tamanho da memória for maior que o tamanho do lote
        # if len(self.memory) > self.batch_size:
        average_Q = 0
        max_Q = -inf
        average_critic_loss = 0
        average_actor_loss = 0

        # Percorrer o número de iterações
        for i in range(n_iteraion):
            print("learn: ", i)
            # ---------------------------- Amostragem de Experiência ---------------------------- #
            # Obtero o índice da amostra, a amostra (s, a, r, s', d) e os pesos de importância
            idxs, state, action, reward, next_state, done, is_weights = (
                self.memory.sample()
            )

            # Converter os pesos de importância para tensor
            is_weights = torch.from_numpy(is_weights).float().to(self.device)

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
                # recompensa + (gamma * Q-valor de destino * (1 - feito))
                Q_targets = (
                    reward
                    + (self.gamma**self.nstep * Q_targets_next * (1 - done)).detach()
                )

            # Passar o estado atual e ação para o modelo crítico
            Q1_expected, Q2_expected = self.critic(state, action)
            # Valor Q esperado minimo
            Q_expected = torch.min(Q1_expected, Q2_expected)
            # Erro absoluto entre os valores Q esperados e os valores Q alvos em np.adarray
            errors = np.abs((Q_expected - Q_targets).detach().cpu().numpy())
            # Calcular a perda do crítico
            critic_loss = F.mse_loss(Q1_expected, Q_targets) + F.mse_loss(
                Q2_expected, Q_targets
            )

            # Atualizar os pesos de importância na memória priorizada
            self.memory.batch_update(idxs, errors)

            # Minimize a perda
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = 0

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

            average_critic_loss += critic_loss
            average_actor_loss += actor_loss

        loss_critic = average_critic_loss / n_iteraion
        loss_actor = average_actor_loss / n_iteraion
        average_policy = average_Q / n_iteraion
        max_policy = max_Q

        return (loss_critic, loss_actor, average_policy, max_policy, True)

        # else:
        #     return (0, 0, 0, 0, False)

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

    @staticmethod
    def save(model: nn.Module, path: str, filename: str, version: str) -> None:
        """Salvar o modelo"""
        torch.save(
            model.state_dict(), path + "checkpoint_" + filename + "_" + version + ".pth"
        )

    @staticmethod
    def load(model: nn.Module, path: str, device: str) -> None:
        """Carregar o modelo"""
        model.load_state_dict(torch.load(path, map_location=device))  # del torch.load
