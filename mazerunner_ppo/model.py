from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MazeFeaturesExtractor(BaseFeaturesExtractor):
    """Small CNN for the symbolic grid plus an MLP for scalar state."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        grid_shape = observation_space["grid"].shape
        stats_dim = observation_space["stats"].shape[0]
        channels = grid_shape[0]

        self.grid_encoder = th.nn.Sequential(
            th.nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            th.nn.ReLU(),
            th.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )
        with th.no_grad():
            sample = th.zeros((1, *grid_shape), dtype=th.float32)
            grid_dim = int(self.grid_encoder(sample).shape[1])

        self.stats_encoder = th.nn.Sequential(
            th.nn.Linear(stats_dim, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 64),
            th.nn.ReLU(),
        )
        self.fusion = th.nn.Sequential(
            th.nn.Linear(grid_dim + 64, features_dim),
            th.nn.ReLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        grid_features = self.grid_encoder(observations["grid"].float())
        stats_features = self.stats_encoder(observations["stats"].float())
        return self.fusion(th.cat((grid_features, stats_features), dim=1))
