# import copy
# import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from components.icm import ICM, Forward, Inverse
from components.replaybuffer import PER
from engine.sanity import seed_everything
from network.model import Actor, Critic
from numpy import inf
from omegaconf import OmegaConf

# Verificar se o cuda está disponivel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """Treinamento do agente com o Ambiente."""

    def __init__(
        self,
        state_size: int = 24,
        action_size: int = 2,
        max_action: int = 1,
        min_action: int = 1,
        noise: float = 0.2,
        noise_std: float = 0.1,
        noise_clip: float = 0.5,
        pretraining: bool = False,
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
        self.agent_config = OmegaConf.load("config/agent.yaml")
        # Setar o seed de todo o ambiente (padrão: 42)
        seed_everything()

        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.min_action = min_action
        self.noise = noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip

        # Parâmetros do ruído
        self.distances = []
        self.desired_distance = 0.7
        self.scalar = 0.05
        self.scalar_decay = 0.99

        # Normalizar o ruído
        self.normal_scalar = 0.25
        self.nstep = 10
        self.eta = 0.1

        # Transfência de aprendizado
        if pretraining:

            # Rede "Actor" (com/ Rede Alvo)
            self.actor = Actor(state_size, action_size, float(self.max_action)).to(
                device
            )
            self.actor.load_state_dict(torch.load("/content/checkpoint_actor.pth"))
            # Definir o modelo alvo, porém, sem a necessidade de calcular gradientes
            self.actor_target = (
                Actor(state_size, action_size, float(self.max_action))
                .to(device)
                .eval()
                .requires_grad_(False)
            )
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=self.agent_config.MODEL.LR_ACTOR
            )
            self.actor_optimizer.load_state_dict(
                torch.load("/content/checkpoint_actor_optimizer.pth")
            )

            # Rede "Actor" para ruído
            self.actor_noised = Actor(
                state_size, action_size, float(self.max_action)
            ).to(device)

            # Rede "Actor" (com/ Rede Alvo)
            self.critic = Critic(state_size, action_size).to(device)
            self.critic.load_state_dict(
                torch.load("/content/checkpoint_critic.pth", map_location=device)
            )
            self.critic_target = (
                Critic(state_size, action_size).to(device).eval().requires_grad_(False)
            )
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=self.agent_config.MODEL.LR_CRITIC
            )
            self.critic_optimizer.load_state_dict(
                torch.load("/content/checkpoint_critic_optimizer.pth")
            )

        else:
            # Rede "Actor" (com/ Rede Alvo)
            self.actor = Actor(state_size, action_size, float(max_action)).to(device)
            self.actor_target = (
                Actor(state_size, action_size, float(max_action))
                .to(device)
                .eval()
                .requires_grad_(False)
            )
            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=self.agent_config.MODEL.LR_ACTOR
            )

            self.actor_noised = Actor(state_size, action_size, float(max_action)).to(
                device
            )

            # Rede "Actor" (com/ Rede Alvo)
            self.critic = Critic(state_size, action_size).to(device)
            self.critic_target = (
                Critic(state_size, action_size).to(device).eval().requires_grad_(False)
            )
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=self.agent_config.MODEL.LR_CRITIC
            )

        self.clip_grad = torch.nn.utils.clip_grad_norm_

        # Memória de replay priorizada
        # Fonte: https://arxiv.org/abs/1511.05952
        self.memory = PER(
            self.agent_config.RL.BUFFER_SIZE,
            self.agent_config.MODEL.BATCH_SIZE,
            self.agent_config.RL.GAMMA,
            self.nstep,
        )

        # Inicializar o modelo de Inverso e Direto
        # Fonte: https://arxiv.org/abs/1705.05363
        inverse_m = Inverse(self.state_size, self.action_size)
        forward_m = Forward(
            self.state_size,
            self.action_size,
            inverse_m.calc_input_layer(),
            device=device,
        )
        # Inicializar o modelo de ICM
        self.ICM = ICM(inverse_m, forward_m).to(device)

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if isinstance(state, tuple):
            state = np.array(state[0])
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
            states.shape[0] == 24
        ), "O Tamanho dos estados não é (24) em PREDICT -> states size: {}.".format(
            states.shape[0]
        )
        assert (
            states.ndim == 1
        ), "O ndim dos estados não é (1) em PREDICT -> estados ndim: {}.".format(
            states.ndim
        )

        # Verificar se o estado não é uma tupla, se for converter para (np.array)
        if isinstance(states, tuple):
            states = np.array(states[0])

        # Converter estados para tensor
        state = torch.from_numpy(states).float().to(device)

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
        self, n_iteraion: int, episode: int, gamma: float = GAMMA
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Atualize parâmetros de política e valor usando determinado lote de tuplas de experiência.

        Parâmetros
        ======
            n_iteraion (int): O número de iterações para treinar a rede
            gamma (float): Factor de desconto
        """

        if episode % 200 == 0:
            self.save(self.actor, "/content/", "actor", str(episode))
            self.save(self.critic, "/content/", "critic", str(episode))

        self.actor.train()
        self.critic.train()

        # Se o tamanho da memória for maior que o tamanho do lote
        if len(self.memory) > self.agent_config.MODEL.BATCH_SIZE:
            average_Q = 0
            max_Q = -inf
            average_critic_loss = 0
            average_actor_loss = 0

            # Percorrer o número de iterações
            for i in range(n_iteraion):
                # ---------------------------- Amostragem de Experiência ---------------------------- #
                # Obtero o índice da amostra, a amostra (s, a, r, s', d) e os pesos de importância
                idxs, (state, action, reward, next_state, done), is_weights = (
                    self.memory.sample()
                )

                # Converter os pesos de importância para tensor
                is_weights = torch.from_numpy(is_weights).float().to(device)

                # Converter a ação para tensor
                action_ = action.cpu().numpy()

                # Calcula o erro de predição do modelo de inversão e avanço
                forward_pred_err, inverse_pred_err = self.ICM.calc_errors(
                    state1=state, state2=next_state, action=action
                )
                # Calcula a recompensa intrínseca
                reward_intrinsic = self.eta * forward_pred_err
                assert (
                    reward_intrinsic.shape == reward.shape
                ), "recompensa e recompensa intrínseca não têm a mesma forma"
                # Adiciona a recompensa intrínseca à recompensa
                reward += reward_intrinsic.detach()

                # ---------------------------- Atualizar crítico ---------------------------- #
                # Obtenha ações de próximo estado previstas e valores Q de modelos de destino

                # Desativar o cálculo de gradientes
                with torch.no_grad():

                    # Gerar um ruído aleatório para a ação
                    noise = (
                        torch.FloatTensor(action_)
                        .data.normal_(0, self.noise)
                        .to(device)
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
                    Q_targets = reward + (gamma * Q_targets_next * (1 - done)).detach()

                # Passar o estado atual e ação para o modelo crítico
                Q1_expected, Q2_expected = self.critic(state, action)
                # Valor Q esperado minimo
                Q_expected = torch.min(Q1_expected, Q2_expected)
                # Erro absoluto entre os valores Q esperados e os valores Q alvos em np.adarray
                errors = np.abs((Q_expected - Q_targets).detach().cpu().numpy())
                # Erro do modelo de inversão e avanço
                icm_loss = self.ICM.update_ICM(forward_pred_err, inverse_pred_err)
                print(icm_loss)
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

                if i % self.agent_config.RL.UPDATE_EVERY_STEP == 0:
                    # ---------------------------- atualizar Ator ---------------------------- #
                    # Calcular perda de ator
                    actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                    # Minimize a perda
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ----------------------- Atualizar redes de destino ----------------------- #
                    # Passar todos os pesos da rede de destino para a rede local usando a atualização suave
                    self.soft_update(
                        self.critic, self.critic_target, self.agent_config.RL.TAU
                    )
                    self.soft_update(
                        self.actor, self.actor_target, self.agent_config.RL.TAU
                    )

                average_critic_loss += critic_loss
                average_actor_loss += actor_loss

            loss_critic = average_critic_loss / n_iteraion
            loss_actor = average_actor_loss / n_iteraion
            average_policy = average_Q / n_iteraion
            max_policy = max_Q

            return (loss_critic, loss_actor, average_policy, max_policy)

        else:
            return (0, 0, 0, 0)

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
        torch.save(model.state_dict(), path + "_" + filename + "_" + version + ".pth")

    @staticmethod
    def load(model: nn.Module, filename: str, device: str) -> None:
        """Carregar o modelo"""
        model.load_state_dict(
            torch.load(filename + "_critic.pth", map_location=device)
        )  # del torch.load
