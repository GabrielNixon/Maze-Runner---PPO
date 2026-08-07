from __future__ import annotations

import numpy as np
import pytest

from mazerunner_ppo.exact_env import (
    ExactBackendUnavailable,
    ExactGameConfig,
    ExactMazeRunnerEnv,
    find_exact_library,
)


def require_library():
    try:
        return find_exact_library()
    except ExactBackendUnavailable:
        pytest.skip("exact backend has not been built")


def rollout(seed: int) -> tuple[list[np.ndarray], list[dict]]:
    path = require_library()
    env = ExactMazeRunnerEnv(
        config=ExactGameConfig(frame_skip=3, max_episode_seconds=5.0),
        curriculum_level=3,
        library_path=path,
    )
    observations = []
    infos = []
    try:
        obs, info = env.reset(seed=seed)
        observations.append(np.concatenate([obs["grid"].ravel(), obs["stats"]]))
        infos.append(info)
        for action in [1, 3, 5, 7, 2, 4, 6, 8, 0] * 2:
            obs, _, terminated, truncated, info = env.step(action)
            observations.append(np.concatenate([obs["grid"].ravel(), obs["stats"]]))
            infos.append(info)
            if terminated or truncated:
                break
    finally:
        env.close()
    return observations, infos


def test_exact_observation_contract() -> None:
    path = require_library()
    env = ExactMazeRunnerEnv(library_path=path)
    try:
        obs, info = env.reset(seed=42)
        assert obs["grid"].shape == (7, 19, 19)
        assert obs["stats"].shape == (14,)
        assert obs["grid"].dtype == np.float32
        assert obs["stats"].dtype == np.float32
        assert env.observation_space.contains(obs)
        assert info["backend"] == "exact-c"
    finally:
        env.close()


def test_exact_seed_is_deterministic() -> None:
    first_obs, first_info = rollout(1234)
    second_obs, second_info = rollout(1234)
    assert len(first_obs) == len(second_obs)
    for left, right in zip(first_obs, second_obs, strict=True):
        np.testing.assert_array_equal(left, right)
    assert first_info == second_info


def test_one_exact_env_per_process() -> None:
    path = require_library()
    first = ExactMazeRunnerEnv(library_path=path)
    try:
        with pytest.raises(RuntimeError, match="one ExactMazeRunnerEnv"):
            ExactMazeRunnerEnv(library_path=path)
    finally:
        first.close()
