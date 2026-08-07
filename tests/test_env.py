from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from mazerunner_ppo.env import MazeRunnerConfig, MazeRunnerEnv


def test_gymnasium_contract() -> None:
    env = MazeRunnerEnv(curriculum_level=3)
    check_env(env, skip_render_check=True)
    env.close()


def test_observation_shapes_and_ranges() -> None:
    env = MazeRunnerEnv(curriculum_level=3)
    obs, info = env.reset(seed=123)
    assert obs["grid"].shape == (7, 19, 19)
    assert obs["stats"].shape == (14,)
    assert obs["grid"].dtype == np.float32
    assert obs["stats"].dtype == np.float32
    assert np.all((obs["grid"] >= 0.0) & (obs["grid"] <= 1.0))
    assert np.all((obs["stats"] >= 0.0) & (obs["stats"] <= 1.0))
    assert info["survival_steps"] == 0
    env.close()


def test_seeded_reset_is_deterministic() -> None:
    env_a = MazeRunnerEnv(curriculum_level=3)
    env_b = MazeRunnerEnv(curriculum_level=3)
    obs_a, _ = env_a.reset(seed=44)
    obs_b, _ = env_b.reset(seed=44)
    np.testing.assert_array_equal(obs_a["grid"], obs_b["grid"])
    np.testing.assert_array_equal(obs_a["stats"], obs_b["stats"])

    actions = [1, 3, 5, 7, 0, 2, 4]
    for action in actions:
        out_a = env_a.step(action)
        out_b = env_b.step(action)
        np.testing.assert_array_equal(out_a[0]["grid"], out_b[0]["grid"])
        np.testing.assert_array_equal(out_a[0]["stats"], out_b[0]["stats"])
        assert out_a[1:] == out_b[1:]
    env_a.close()
    env_b.close()


def test_episode_truncates_at_limit_without_hazards() -> None:
    config = MazeRunnerConfig(max_episode_steps=5)
    env = MazeRunnerEnv(config=config, curriculum_level=0)
    env.reset(seed=9)
    truncated = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
    assert truncated
    env.close()


def test_rgb_render() -> None:
    env = MazeRunnerEnv(render_mode="rgb_array", curriculum_level=3)
    env.reset(seed=3)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    env.close()
