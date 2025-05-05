import torch
import torch.nn as nn
from torch.distributions import Categorical


class FlexMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int]):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


def _map_sb3_keys(sd: dict) -> dict:
    new = {}
    for k, v in sd.items():
        if k.startswith("mlp_extractor.policy_net."):
            new_k = k.replace("mlp_extractor.policy_net.", "backbone.body.")
        elif k.startswith("action_net."):
            new_k = k.replace("action_net.", "head.")
        else:
            continue
        new[new_k] = v
    return new


class RNLPolicy(nn.Module):
    def __init__(self, in_dim: int, n_act: int,
                 hidden: list[int], pth: str, device: str = "cpu"):
        super().__init__()
        self.backbone = FlexMLP(in_dim, hidden)
        self.head = nn.Linear(hidden[-1], n_act)

        ckpt = torch.load(pth, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        ckpt = _map_sb3_keys(ckpt)           # renomeia
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        assert not missing, f"Pesos faltando: {missing}"
        self.eval()

    @torch.no_grad()
    def act(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits = self.head(self.backbone(obs))
        return Categorical(logits=logits).sample().item()
