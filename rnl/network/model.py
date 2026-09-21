from typing import Callable, Sequence, Tuple

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class CustomNetwork(nn.Module):
    """
    MLP extractor that lets you pick the *hidden* layer sizes
    while mantendo latent_dim_pi / latent_dim_vf = 32.
    """

    def __init__(self, feature_dim: int, hidden: Sequence[int] = (128, 128, 64)):
        super().__init__()

        self.latent_dim_pi = 32  # tamanho final fixo
        self.latent_dim_vf = 32

        def block(in_f: int, out_f: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.LayerNorm(out_f),
                nn.LeakyReLU(),
            )

        # build stacks dinamicamente
        policy_layers = []
        value_layers = []
        in_dim = feature_dim
        for h in hidden:
            policy_layers.append(block(in_dim, h))
            value_layers.append(block(in_dim, h))
            in_dim = h

        # última projeção para 32
        policy_layers.append(block(in_dim, self.latent_dim_pi))
        value_layers.append(block(in_dim, self.latent_dim_vf))

        self.policy_net = nn.Sequential(*policy_layers)
        self.value_net = nn.Sequential(*value_layers)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Policy que aceita `hidden_sizes` via policy_kwargs
    (ex.: policy_kwargs=dict(hidden_sizes=(256,128,64))).
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        hidden_sizes: Sequence[int] = (128, 128, 64),
        **kwargs,
    ):
        kwargs["ortho_init"] = False
        self.hidden_sizes = hidden_sizes
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # constrói o extractor usando o tamanho escolhido
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(
            feature_dim=self.features_extractor.features_dim,
            hidden=self.hidden_sizes,
        )
