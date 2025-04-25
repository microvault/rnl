import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

class AttnNet(nn.Module):
    def __init__(self, feature_dim: int, heads: int = 2, embed: int = 20):
        super().__init__()
        self.latent_dim_pi = embed
        self.latent_dim_vf = embed

        self.embed = nn.Linear(feature_dim, embed)
        self.attn  = nn.MultiheadAttention(embed, heads, batch_first=True)

        self.pi = nn.Sequential(nn.Linear(embed, embed), nn.ReLU())
        self.vf = nn.Sequential(nn.Linear(embed, embed), nn.ReLU())

    def _attn(self, x):
        z = th.relu(self.embed(x)).unsqueeze(1)
        z, _ = self.attn(z, z, z)
        return z.squeeze(1)

    def forward_actor (self, f): return self.pi(self._attn(f))
    def forward_critic(self, f): return self.vf(self._attn(f))

    def forward(self, f):
        return self.forward_actor(f), self.forward_critic(f)


class AttnPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 lr_schedule, **kw):
        kw["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kw)

    def _build_mlp_extractor(self):
        feat_dim = self.features_extractor.features_dim
        self.mlp_extractor = AttnNet(feat_dim)
