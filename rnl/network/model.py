from typing import Sequence, Tuple

import torch as th
from torch import nn


class CustomNetwork(nn.Module):
    """
    MLP extractor that lets you pick the *hidden* layer sizes
    while keeping latent_dim_pi / latent_dim_vf = 32.
    """

    def __init__(self, feature_dim: int, hidden: Sequence[int] = (128, 128, 64)):
        super().__init__()

        self.latent_dim_pi = 32  # fixed final size
        self.latent_dim_vf = 32

        def block(in_f: int, out_f: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.LayerNorm(out_f),
                nn.LeakyReLU(),
            )

        # build stacks dynamically
        policy_layers = []
        value_layers = []
        in_dim = feature_dim
        for h in hidden:
            policy_layers.append(block(in_dim, h))
            value_layers.append(block(in_dim, h))
            in_dim = h

        # final projection to 32
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


class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic policy network for PPO.
    Policy that accepts `hidden_sizes` via constructor
    (ex.: hidden_sizes=(256,128,64)).
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128, 64),
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        # Feature extractor (identity in this case)
        self.features_dim = observation_dim

        # MLP extractor
        self.mlp_extractor = CustomNetwork(
            feature_dim=self.features_dim,
            hidden=self.hidden_sizes,
        )

        # Action distribution head
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim)

        # Value function head
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Forward pass of the policy.
        Returns: (action_logits, values)
        """
        # Extract features (identity in this case)
        features = obs

        # Get latent representations
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Get action logits and values
        action_logits = self.action_net(latent_pi)
        values = self.value_net(latent_vf)

        return action_logits, values
