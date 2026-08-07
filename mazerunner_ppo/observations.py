from __future__ import annotations

from typing import Any

import numpy as np


def build_observation(env: Any) -> dict[str, np.ndarray]:
    n, (cr, cc) = env.config.observation_size, env.world.center
    half = n // 2
    rs = slice(cr - half, cr + half + 1)
    cs = slice(cc - half, cc + half + 1)
    grid = np.zeros((7, n, n), np.float32)
    grid[0] = env.world.walls[rs, cs]
    grid[1] = env.world.orbs[rs, cs]
    grid[2] = env.world.spikes[rs, cs]
    warning = env.config.spike_warning_start <= env.spike_phase < env.config.spike_up_start
    if env.spikes_up:
        grid[3] = grid[2]
    elif warning:
        grid[4] = grid[2]

    nearest = 1.0
    for enemy in env.enemies.items:
        row, col = enemy.row - cr + half, enemy.col - cc + half
        if 0 <= row < n and 0 <= col < n:
            grid[6 if enemy.frozen else 5, row, col] = 1.0
    if env.enemies.items:
        distance = min(max(abs(e.row - cr), abs(e.col - cc)) for e in env.enemies.items)
        nearest = min(1.0, distance / max(half, 1))

    phase = env.spike_phase / env.config.spike_period_steps
    stats = np.zeros(14, np.float32)
    stats[:5] = (
        env.hunger,
        (np.sin(2 * np.pi * phase) + 1) / 2,
        (np.cos(2 * np.pi * phase) + 1) / 2,
        min(1.0, len(env.enemies.items) / env.config.max_enemies),
        nearest,
    )
    stats[5 + env.last_action] = 1.0
    return {"grid": grid, "stats": stats}


def build_info(env: Any) -> dict[str, Any]:
    return {
        "survival_steps": env.steps,
        "survival_seconds": env.steps * env.config.decision_dt,
        "orbs_collected": env.orbs,
        "hunger": env.hunger,
        "enemy_count": len(env.enemies.items),
        "spike_phase": env.spike_phase,
        "death_reason": env.death_reason,
        "curriculum_level": env.curriculum_level,
    }
