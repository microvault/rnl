from typing import Callable, Tuple, Type
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        h1: int,
        h2: int,
        activation_fn: Type[nn.Module],
    ):
        super().__init__()
        act = activation_fn()
        def block(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.LayerNorm(out_f),
                act,
            )

        self.policy_net = nn.Sequential(
            block(feat_dim, h1),
            block(h1, h2),
        )
        self.value_net = nn.Sequential(
            block(feat_dim, h1),
            block(h1, h2),
        )

        self.latent_dim_pi = h2
        self.latent_dim_vf = h2

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        obs_space: spaces.Space,
        act_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        h1: int,
        h2: int,
        activation_fn: Type[nn.Module],
        **kwargs,
    ):
        self.h1 = h1
        self.h2 = h2
        self.activation_fn = activation_fn
        kwargs['ortho_init'] = False
        super().__init__(obs_space, act_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(
            feat_dim=self.features_extractor.features_dim,
            h1=self.h1,
            h2=self.h2,
            activation_fn=self.activation_fn,
        )
