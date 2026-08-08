from __future__ import annotations

import heapq
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

# Actions match the exact C bridge / browser controller.
_ACTION_DELTAS = {
    0: (0, 0),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, 1),
    4: (1, 1),
    5: (1, 0),
    6: (1, -1),
    7: (0, -1),
    8: (-1, -1),
}


@dataclass(frozen=True)
class ForagingRewardConfig:
    """Training-only shaping for food timing, path planning, and spike awareness."""

    path_scale: float = 0.45
    idle_path_fraction: float = 0.20
    useful_orb_bonus: float = 3.0
    action_spike_penalty: float = 1.25
    center_spike_penalty: float = 0.40
    spike_hit_penalty: float = 3.0
    safe_spike_cost: float = 0.15
    warning_spike_cost: float = 3.5
    active_spike_cost: float = 8.0

    def validate(self) -> None:
        non_negative = {
            "path_scale": self.path_scale,
            "useful_orb_bonus": self.useful_orb_bonus,
            "action_spike_penalty": self.action_spike_penalty,
            "center_spike_penalty": self.center_spike_penalty,
            "spike_hit_penalty": self.spike_hit_penalty,
            "safe_spike_cost": self.safe_spike_cost,
            "warning_spike_cost": self.warning_spike_cost,
            "active_spike_cost": self.active_spike_cost,
        }
        for name, value in non_negative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.idle_path_fraction <= 1.0:
            raise ValueError("idle_path_fraction must be between zero and one")
        if self.warning_spike_cost < self.safe_spike_cost:
            raise ValueError("warning_spike_cost must be >= safe_spike_cost")
        if self.active_spike_cost < self.warning_spike_cost:
            raise ValueError("active_spike_cost must be >= warning_spike_cost")


def _spike_cost(
    grid: np.ndarray,
    row: int,
    col: int,
    *,
    safe_cost: float,
    warning_cost: float,
    active_cost: float,
) -> float:
    if grid.shape[0] >= 4 and grid[3, row, col] > 0.5:
        return active_cost
    if grid.shape[0] >= 5 and grid[4, row, col] > 0.5:
        return warning_cost
    if grid.shape[0] >= 3 and grid[2, row, col] > 0.5:
        return safe_cost
    return 0.0


def nearest_reachable_orb_distance(
    grid: np.ndarray,
    *,
    safe_spike_cost: float = 0.15,
    warning_spike_cost: float = 3.5,
    active_spike_cost: float = 8.0,
) -> float:
    """Return normalized hazard-aware shortest path cost to a visible orb.

    Channel 0 is walls, 1 is orbs, 2 is any spike tile, 3 is active spikes,
    and 4 is warning spikes. Unlike the original Chebyshev shaping, this route
    cannot pass through walls or diagonal corners and it prefers routes that do
    not cross spikes which are warning or currently active.

    A value of 1.0 means no visible orb is reachable from the observation center.
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
    queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    best = {start: 0.0}

    while queue:
        cost, row, col = heapq.heappop(queue)
        if cost > best[(row, col)]:
            continue
        if orbs[row, col] and (row, col) != start:
            normalizer = float(max(height, width, 1))
            return float(min(1.0, cost / normalizer))

        for dy, dx in directions:
            next_row, next_col = row + dy, col + dx
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if walls[next_row, next_col]:
                continue
            if dy != 0 and dx != 0:
                if walls[row + dy, col] or walls[row, col + dx]:
                    continue

            hazard = _spike_cost(
                grid,
                next_row,
                next_col,
                safe_cost=safe_spike_cost,
                warning_cost=warning_spike_cost,
                active_cost=active_spike_cost,
            )
            next_cost = cost + 1.0 + hazard
            key = (next_row, next_col)
            if next_cost < best.get(key, float("inf")):
                best[key] = next_cost
                heapq.heappush(queue, (next_cost, next_row, next_col))

    return 1.0


def useful_orb_fraction(hunger_before: float) -> float:
    """Fraction of an orb's +0.5 hunger refill that can actually be used."""

    room = max(0.0, 1.0 - float(hunger_before))
    return float(min(1.0, room / 0.5))


def spike_risk_at(grid: np.ndarray, row: int, col: int) -> float:
    """Risk score for occupying one observed tile at the current spike phase."""

    if not (0 <= row < grid.shape[1] and 0 <= col < grid.shape[2]):
        return 0.0
    if grid.shape[0] >= 4 and grid[3, row, col] > 0.5:
        return 1.0
    if grid.shape[0] >= 5 and grid[4, row, col] > 0.5:
        return 0.55
    if grid.shape[0] >= 3 and grid[2, row, col] > 0.5:
        return 0.05
    return 0.0


def action_spike_risk(grid: np.ndarray, action: int) -> float:
    """Risk of the tile selected by an action, using only the visible grid."""

    if action not in _ACTION_DELTAS:
        raise ValueError(f"invalid action {action}")
    center_row = grid.shape[1] // 2
    center_col = grid.shape[2] // 2
    dy, dx = _ACTION_DELTAS[action]
    return spike_risk_at(grid, center_row + dy, center_col + dx)


class ForagingRewardWrapper(gym.Wrapper):
    """Dense skill shaping while leaving game dynamics and observations unchanged.

    The wrapper trains three behaviors that the raw survival reward underspecifies:
    route to reachable food, delay pickups until the refill is useful, and react to
    warning/active spike channels before actual damage occurs.
    """

    def __init__(self, env: gym.Env, config: ForagingRewardConfig | None = None) -> None:
        super().__init__(env)
        self.foraging_config = config or ForagingRewardConfig()
        self.foraging_config.validate()
        self._previous_grid: np.ndarray | None = None
        self._previous_path_distance = 1.0
        self._previous_hunger = 1.0
        self._previous_orbs = 0
        self._estimated_spike_hits = 0
        self._active_spike_choices = 0
        self._warning_spike_choices = 0
        self._useful_orb_total = 0.0

    def _path_distance(self, grid: np.ndarray) -> float:
        return nearest_reachable_orb_distance(
            grid,
            safe_spike_cost=self.foraging_config.safe_spike_cost,
            warning_spike_cost=self.foraging_config.warning_spike_cost,
            active_spike_cost=self.foraging_config.active_spike_cost,
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._previous_grid = observation["grid"].copy()
        self._previous_path_distance = self._path_distance(observation["grid"])
        self._previous_hunger = float(info["hunger"])
        self._previous_orbs = int(info["orbs_collected"])
        self._estimated_spike_hits = 0
        self._active_spike_choices = 0
        self._warning_spike_choices = 0
        self._useful_orb_total = 0.0
        return observation, self._augment_info(info, None)

    def _augment_info(self, info: dict, useful_fraction: float | None) -> dict:
        result = dict(info)
        orb_count = max(int(result.get("orbs_collected", 0)), 1)
        result["skill_path_distance"] = self._previous_path_distance
        result["skill_useful_orb_fraction"] = useful_fraction
        result["skill_orb_refill_efficiency"] = self._useful_orb_total / orb_count
        result["skill_spike_hits"] = self._estimated_spike_hits
        result["skill_active_spike_choices"] = self._active_spike_choices
        result["skill_warning_spike_choices"] = self._warning_spike_choices
        return result

    def step(self, action):
        action_int = int(action)
        pre_risk = (
            action_spike_risk(self._previous_grid, action_int)
            if self._previous_grid is not None
            else 0.0
        )
        if pre_risk >= 0.99:
            self._active_spike_choices += 1
        elif pre_risk >= 0.5:
            self._warning_spike_choices += 1

        observation, reward, terminated, truncated, info = self.env.step(action)
        current_grid = observation["grid"]
        current_distance = self._path_distance(current_grid)
        current_hunger = float(info["hunger"])
        current_orbs = int(info["orbs_collected"])
        orb_delta = max(0, current_orbs - self._previous_orbs)

        # Food routing: a small proactive pull keeps food locations salient while
        # full, then the potential becomes much stronger as hunger falls.
        if orb_delta == 0:
            urgency = (1.0 - current_hunger) ** 2
            path_weight = self.foraging_config.path_scale * (
                self.foraging_config.idle_path_fraction
                + (1.0 - self.foraging_config.idle_path_fraction) * urgency
            )
            reward += path_weight * (self._previous_path_distance - current_distance)

        useful_fraction = useful_orb_fraction(self._previous_hunger)
        if orb_delta:
            self._useful_orb_total += useful_fraction * float(orb_delta)
            reward += (
                self.foraging_config.useful_orb_bonus
                * useful_fraction
                * float(orb_delta)
            )

        # Spike awareness: penalize choosing a tile that is already warning/up,
        # and also penalize lingering on one after the step. This gives a dense
        # teaching signal before the sparse 0.5-hunger damage event.
        reward -= self.foraging_config.action_spike_penalty * pre_risk
        center = current_grid.shape[1] // 2
        center_risk = spike_risk_at(current_grid, center, center)
        reward -= self.foraging_config.center_spike_penalty * center_risk

        # A spike hit creates an abrupt ~0.5 hunger drop. Normal decay over a
        # 0.1-second decision is only ~0.01, so this is a robust training metric.
        hunger_drop = self._previous_hunger - current_hunger
        spike_hit = orb_delta == 0 and hunger_drop > 0.25
        if spike_hit:
            self._estimated_spike_hits += 1
            reward -= self.foraging_config.spike_hit_penalty

        self._previous_grid = current_grid.copy()
        self._previous_path_distance = current_distance
        self._previous_hunger = current_hunger
        self._previous_orbs = current_orbs
        return (
            observation,
            float(reward),
            terminated,
            truncated,
            self._augment_info(info, useful_fraction if orb_delta else None),
        )
