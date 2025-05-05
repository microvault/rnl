from typing import Callable, Tuple
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.latent_dim_pi = 16
        self.latent_dim_vf = 16

        def block(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.LayerNorm(out_f),
                nn.LeakyReLU(),
            )

        self.policy_net = nn.Sequential(
            block(feature_dim, 32),
            block(32, 32),
            block(32, 16),
        )

        self.value_net = nn.Sequential(
            block(feature_dim, 32),
            block(32, 32),
            block(32, 16),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        **kwargs,
    ):
        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(
            feature_dim=self.features_extractor.features_dim
        )
