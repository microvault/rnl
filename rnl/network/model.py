from typing import Callable, Tuple

import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_pi),
            nn.ReLU()
        )

        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_vf),
            nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomNetworkLSTM(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        hidden_size: int = 64
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.lstm = nn.LSTM(feature_dim, hidden_size, batch_first=True)

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size, last_layer_dim_pi),
            nn.ReLU()
        )
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, last_layer_dim_vf),
            nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        x = features.unsqueeze(1)  # (batch, seq=1, feature_dim)
        _, (h, _) = self.lstm(x)
        return self.policy_net(h[-1])

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        x = features.unsqueeze(1)
        _, (h, _) = self.lstm(x)
        return self.value_net(h[-1])


class CustomActorCriticPolicy(ActorCriticPolicy):
    """Policy para usar a rede Linear."""
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor_
