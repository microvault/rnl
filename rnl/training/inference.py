import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

def clamp(value, vmin, vmax):
    return max(vmin, min(value, vmax))


def distance_to_goal(x, y, goal_x, goal_y):
    return min(np.hypot(x - goal_x, y - goal_y), 9.0)


def angle_to_goal(x, y, theta, goal_x, goal_y):
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])
    return float(abs(np.arctan2(np.cross(o_t, g_t), np.dot(o_t, g_t))))


class FlexMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


def _map_sb3_keys(sd):
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
    def __init__(self, in_dim, n_act, hidden, pth, device="cpu"):
        super().__init__()
        self.backbone = FlexMLP(in_dim, hidden)
        self.head = nn.Linear(hidden[-1], n_act)

        ckpt = torch.load(pth, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = _map_sb3_keys(ckpt)
        self.load_state_dict(ckpt, strict=False)
        self.eval()

    @torch.no_grad()
    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return Categorical(logits=self.head(self.backbone(obs))).sample().item()
