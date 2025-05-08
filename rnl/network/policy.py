import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple

_ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
}

def get_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(f"Ativação '{name}' não suportada")

def block(in_f: int, out_f: int, act: nn.Module) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.LayerNorm(out_f),
        act,
    )

class PolicyBackbone(nn.Module):
    def __init__(self, feature_dim: int, hidden_sizes: Tuple[int,int,int], activation: str):
        super().__init__()
        act = get_activation(activation)
        h1, h2, h3 = hidden_sizes
        self.body = nn.Sequential(
            block(feature_dim, h1, act),
            block(h1, h2, act),
            block(h2, h3, act),
        )
        self.latent_dim = h3

class RNLPolicy(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_act: int,
        pth: str,
        hidden_sizes: Tuple[int,int,int],
        activation: str,
    ):
        super().__init__()
        self.backbone = PolicyBackbone(in_dim, hidden_sizes, activation)
        self.head     = nn.Linear(self.backbone.latent_dim, n_act)

        # carrega checkpoint
        ckpt = torch.load(pth, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        # mapeia chaves do SB3
        new_sd = {}
        for k, v in ckpt.items():
            if k.startswith("mlp_extractor.policy_net."):
                new_sd[k.replace("mlp_extractor.policy_net.", "backbone.body.")] = v
            elif k.startswith("action_net."):
                new_sd[k.replace("action_net.", "head.")] = v

        # carrega pesos
        self.load_state_dict(new_sd, strict=False)
        self.eval()

    @torch.no_grad()
    def act(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        feats  = self.backbone.body(obs)
        logits = self.head(feats)
        return Categorical(logits=logits).sample().item()
