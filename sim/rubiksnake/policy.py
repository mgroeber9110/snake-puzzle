"""Policy network that follows the chain structure of the snake.

Every joint gets a feature vector (its angle, target angle, difference,
which of its two moves are allowed, which were found blocked and which
would lead back to a shape visited earlier in the episode), and a stack of dilated 1D convolutions
along the chain mixes information between joints; with dilations up to 8 every
joint sees the whole snake. The two logits of a joint (turn +1 / -1) come from
the same small head for every joint, so what is learned for one joint carries
over to all others.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class _Block(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        return x + self.mix(torch.relu(self.conv(torch.relu(x))))


class ChainExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, channels: int = 64,
                 dilations=(1, 2, 4, 8, 1, 2, 4, 8)):
        self.num_joints = len(observation_space["angles"].nvec)
        super().__init__(observation_space, features_dim=self.num_joints * channels)
        self.keys = [k for k in ("angles", "target", "delta") if k in observation_space.spaces]
        # Per-move flags, two channels per joint (turn +1 / -1), in this order.
        self.flag_keys = [k for k in ("blocked", "revisit") if k in observation_space.spaces]
        in_ch = 4 * len(self.keys) + 2 + 2 * len(self.flag_keys)
        self.inp = nn.Conv1d(in_ch, channels, 1)
        self.blocks = nn.Sequential(*[_Block(channels, d) for d in dilations])
        self.channels = channels

    def forward(self, obs):
        b, n = obs["action_mask"].shape[0], self.num_joints
        parts = [obs[k].view(b, n, 4) for k in self.keys] + [obs["action_mask"].view(b, n, 2)]
        parts += [obs[k].view(b, n, 2) for k in self.flag_keys]
        x = torch.cat(parts, dim=2).transpose(1, 2)            # (B, C_in, joints)
        x = torch.relu(self.blocks(self.inp(x)))                # (B, C, joints)
        return x.transpose(1, 2).reshape(b, n * self.channels)  # joint-major


class PerJointHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 2))

    def forward(self, latent):
        b = latent.shape[0]
        return self.net(latent.view(b, -1, self.channels)).reshape(b, -1)  # (B, 2*joints)


class ChainPolicy(MaskableMultiInputActorCriticPolicy):
    """MaskablePPO policy with ChainExtractor and a shared per-joint action head."""

    def __init__(self, *args, channels: int = 64, **kwargs):
        kwargs.setdefault("features_extractor_class", ChainExtractor)
        kwargs.setdefault("features_extractor_kwargs", {"channels": channels})
        kwargs.setdefault("net_arch", {"pi": [], "vf": [256]})
        self._channels = channels
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = PerJointHead(self._channels)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
