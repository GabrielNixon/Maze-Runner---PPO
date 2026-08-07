from __future__ import annotations

import torch

from mazerunner_ppo.env import MazeRunnerEnv
from mazerunner_ppo.model import MazeFeaturesExtractor


def test_feature_extractor_output_shape() -> None:
    env = MazeRunnerEnv()
    obs, _ = env.reset(seed=1)
    extractor = MazeFeaturesExtractor(env.observation_space, features_dim=128)
    batch = {
        "grid": torch.from_numpy(obs["grid"]).unsqueeze(0),
        "stats": torch.from_numpy(obs["stats"]).unsqueeze(0),
    }
    output = extractor(batch)
    assert output.shape == (1, 128)
    env.close()
