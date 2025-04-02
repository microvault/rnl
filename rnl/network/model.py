from typing import Callable, Tuple

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class CustomNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        # Dimensões de saída (policy e value)
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Transformer para a policy
        self.embed_pi = nn.Linear(feature_dim, d_model)
        encoder_layer_pi = nn.TransformerEncoderLayer(d_model, nhead=nhead, batch_first=True)

        self.transformer_pi = nn.TransformerEncoder(
            encoder_layer_pi, num_layers=num_layers
        )
        self.policy_net = nn.Sequential(
            nn.Linear(d_model, last_layer_dim_pi), nn.ReLU()
        )

        # Transformer para o value
        self.embed_vf = nn.Linear(feature_dim, d_model)
        encoder_layer_vf = nn.TransformerEncoderLayer(d_model, nhead=nhead, batch_first=True)
        self.transformer_vf = nn.TransformerEncoder(
            encoder_layer_vf, num_layers=num_layers
        )
        self.value_net = nn.Sequential(nn.Linear(d_model, last_layer_dim_vf), nn.ReLU())

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # Embedding + Transformer
        x = self.embed_pi(features)  # (batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, seq_len=1, d_model)
        x = self.transformer_pi(x)  # Mesmo shape
        x = x.squeeze(1)  # (batch_size, d_model)
        return self.policy_net(x)  # (batch_size, last_layer_dim_pi)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        x = self.embed_vf(features)
        x = x.unsqueeze(1)
        x = self.transformer_vf(x)
        x = x.squeeze(1)
        return self.value_net(x)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
