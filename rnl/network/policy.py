import torch
import torch.nn as nn
from torch.distributions import Categorical


def block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.ReLU(),
    )


class PolicyBackbone(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            block(feature_dim, 32),
            block(32, 32),
            block(32, 20),
        )


class RNLPolicy(nn.Module):
    def __init__(self, in_dim: int, n_act: int,
                 pth: str, device: str = "cpu"):
        super().__init__()
        self.backbone = PolicyBackbone(in_dim)
        self.head = nn.Linear(20, n_act)

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
        logits = self.head(self.backbone.body(obs))
        return Categorical(logits=logits).sample().item()


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
