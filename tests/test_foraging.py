from __future__ import annotations

import numpy as np

from mazerunner_ppo.foraging import nearest_reachable_orb_distance, useful_orb_fraction


def blank_grid(size: int = 19) -> np.ndarray:
    return np.zeros((7, size, size), dtype=np.float32)


def test_reachable_orb_distance_prefers_actual_path() -> None:
    grid = blank_grid()
    center = grid.shape[1] // 2
    grid[1, center, center + 2] = 1.0
    direct = nearest_reachable_orb_distance(grid)

    # Put a wall directly between player and orb. The orb is still reachable by
    # a detour, so path-aware distance must become longer than the direct case.
    grid[0, center, center + 1] = 1.0
    detour = nearest_reachable_orb_distance(grid)
    assert detour > direct
    assert detour < 1.0


def test_unreachable_orb_returns_one() -> None:
    grid = blank_grid()
    center = grid.shape[1] // 2
    grid[1, center, center + 2] = 1.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                grid[0, center + dy, center + dx] = 1.0
    assert nearest_reachable_orb_distance(grid) == 1.0


def test_useful_orb_fraction_rewards_timing() -> None:
    assert useful_orb_fraction(1.0) == 0.0
    assert useful_orb_fraction(0.75) == 0.5
    assert useful_orb_fraction(0.5) == 1.0
    assert useful_orb_fraction(0.1) == 1.0
