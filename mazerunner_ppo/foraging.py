from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class ForagingRewardConfig:
    path_scale: float = 0.35
    idle_path_fraction: float = 0.25
    useful_orb_bonus: float = 2.0

    def validate(self) -> None:
        if self.path_scale < 0.0:
            raise ValueError("path_scale must be non-negative")
        if not 0.0 <= self.idle_path_fraction <= 1.0:
            raise ValueError("idle_path_fraction must be between zero and one")
        if self.useful_orb_bonus < 0.0:
            raise ValueError("useful_orb_bonus must be non-negative")


def nearest_reachable_orb_distance(grid: np.ndarray) -> float:
    """Return normalized shortest safe path distance to a visible orb.

    Channel 0 is walls and channel 1 is orbs. Movement uses the same eight
    directions available to the policy. Diagonal transitions that would cut
    through a blocked corner are excluded so shaping does not reward paths the
    agent cannot safely execute.

    A value of 1.0 means that no visible orb is reachable from the center.
    """

    if grid.ndim != 3 or grid.shape[0] < 2:
        raise ValueError("grid must contain wall and orb channels")

    walls = grid[0] > 0.5
    orbs = grid[1] > 0.5
    height, width = walls.shape
    start = (height // 2, width // 2)
    if orbs[start]:
        return 0.0
    if walls[start]:
        return 1.0

    directions = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    queue: deque[tuple[int, int, int]] = deque([(start[0], start[1], 0)])
    visited = {start}

    while queue:
        y, x, distance = queue.popleft()
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if walls[ny, nx] or (ny, nx) in visited:
                continue
            if dy != 0 and dx != 0:
                if walls[y + dy, x] or walls[y, x + dx]:
                    continue
            next_distance = distance + 1
            if orbs[ny, nx]:
                normalizer = max(height, width, 1)
                return float(min(1.0, next_distance / normalizer))
            visited.add((ny, nx))
            queue.append((ny, nx, next_distance))

    return 1.0


def useful_orb_fraction(hunger_before: float) -> float:
    """Fraction of an orb's 0.5 hunger refill that can actually be used."""

    room = max(0.0, 1.0 - float(hunger_before))
    return float(min(1.0, room / 0.5))


class ForagingRewardWrapper(gym.Wrapper):
    """Training-only dense reward for reachable, well-timed food collection."""

    def __init__(self, env: gym.Env, config: ForagingRewardConfig | None = None) -> None:
        super().__init__(env)
        self.foraging_config = config or ForagingRewardConfig()
        self.foraging_config.validate()
        self._previous_path_distance = 1.0
        self._previous_hunger = 1.0
        self._previous_orbs = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._previous_path_distance = nearest_reachable_orb_distance(observation["grid"])
        self._previous_hunger = float(info["hunger"])
        self._previous_orbs = int(info["orbs_collected"])
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_distance = nearest_reachable_orb_distance(observation["grid"])
        current_hunger = float(info["hunger"])
        current_orbs = int(info["orbs_collected"])
        orb_delta = max(0, current_orbs - self._previous_orbs)

        # Keep a small proactive pull toward food even when full, then increase
        # it smoothly as hunger falls. Skip potential shaping on pickup frames
        # because the target orb disappears and nearest-target identity changes.
        if orb_delta == 0:
            urgency = (1.0 - current_hunger) ** 2
            path_weight = self.foraging_config.path_scale * (
                self.foraging_config.idle_path_fraction
                + (1.0 - self.foraging_config.idle_path_fraction) * urgency
            )
            reward += path_weight * (self._previous_path_distance - current_distance)

        useful_fraction = useful_orb_fraction(self._previous_hunger)
        if orb_delta:
            reward += (
                self.foraging_config.useful_orb_bonus
                * useful_fraction
                * float(orb_delta)
            )

        info = dict(info)
        info["foraging_path_distance"] = current_distance
        info["foraging_useful_orb_fraction"] = useful_fraction if orb_delta else None

        self._previous_path_distance = current_distance
        self._previous_hunger = current_hunger
        self._previous_orbs = current_orbs
        return observation, float(reward), terminated, truncated, info
