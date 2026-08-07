from __future__ import annotations

import numpy as np

from mazerunner_ppo.config import MazeRunnerConfig


class RollingMap:
    """Procedural tile buffer that discards terrain leaving the window."""

    def __init__(self, cfg: MazeRunnerConfig, rng: np.random.Generator, level: int):
        self.cfg, self.rng, self.level = cfg, rng, level
        n = cfg.buffer_size
        self.walls = np.zeros((n, n), np.bool_)
        self.orbs = np.zeros((n, n), np.bool_)
        self.spikes = np.zeros((n, n), np.bool_)
        self.spawns = np.zeros((n, n), np.bool_)

    @property
    def center(self) -> tuple[int, int]:
        c = self.cfg.buffer_size // 2
        return c, c

    def reset(self) -> None:
        n = self.cfg.buffer_size
        self.walls.fill(True)
        stack = [self.center]
        self.walls[self.center] = False
        directions = ((0, 2), (2, 0), (0, -2), (-2, 0))
        while stack:
            row, col = stack[-1]
            choices = []
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 1 <= nr < n - 1 and 1 <= nc < n - 1 and self.walls[nr, nc]:
                    choices.append((nr, nc, dr, dc))
            if not choices:
                stack.pop()
                continue
            nr, nc, dr, dc = choices[int(self.rng.integers(len(choices)))]
            self.walls[row + dr // 2, col + dc // 2] = False
            self.walls[nr, nc] = False
            stack.append((nr, nc))

        loops = self.rng.random((n, n)) < 0.10
        loops[[0, -1], :] = loops[:, [0, -1]] = False
        self.walls[loops] = False
        self.walls[[0, -1], :] = self.walls[:, [0, -1]] = True
        cr, cc = self.center
        self.walls[cr - 1:cr + 2, cc - 1:cc + 2] = False
        self._populate_all()
        self.orbs[cr, cc] = self.spikes[cr, cc] = self.spawns[cr, cc] = False
        rr, cols = np.ogrid[:n, :n]
        self.spawns[(rr - cr) ** 2 + (cols - cc) ** 2 <= 16] = False

    def _populate_all(self) -> None:
        floor = ~self.walls
        self.orbs[:] = floor & (self.rng.random(self.walls.shape) < self.cfg.orb_density) if self.level >= 1 else False
        self.spikes[:] = floor & (self.rng.random(self.walls.shape) < self.cfg.spike_density) if self.level >= 2 else False
        self.spawns[:] = floor & (self.rng.random(self.walls.shape) < self.cfg.enemy_spawn_density) if self.level >= 3 else False

    def can_move(self, dx: int, dy: int) -> bool:
        if dx == dy == 0:
            return False
        cr, cc = self.center
        if self.walls[cr + dy, cc + dx]:
            return False
        return not (dx and dy and (self.walls[cr, cc + dx] or self.walls[cr + dy, cc]))

    def shift(self, dx: int, dy: int) -> tuple[int, int]:
        sr, sc = -dy, -dx
        for array in (self.walls, self.orbs, self.spikes, self.spawns):
            array[:] = np.roll(array, (sr, sc), axis=(0, 1))
        if dy:
            self._new_row(-1 if dy > 0 else 0)
        if dx:
            self._new_col(-1 if dx > 0 else 0)
        self.walls[self.center] = False
        return sr, sc

    def _new_row(self, row: int) -> None:
        n = self.cfg.buffer_size
        adjacent = n - 2 if row == -1 else 1
        values = self.rng.random(n) < 0.30
        values[[0, -1]] = True
        openings = np.flatnonzero(~self.walls[adjacent, 1:-1]) + 1
        if openings.size:
            values[self.rng.choice(openings, min(3, openings.size), replace=False)] = False
        else:
            values[int(self.rng.integers(1, n - 1))] = False
        self.walls[row] = values
        self._features(row=row)

    def _new_col(self, col: int) -> None:
        n = self.cfg.buffer_size
        adjacent = n - 2 if col == -1 else 1
        values = self.rng.random(n) < 0.30
        values[[0, -1]] = True
        openings = np.flatnonzero(~self.walls[1:-1, adjacent]) + 1
        if openings.size:
            values[self.rng.choice(openings, min(3, openings.size), replace=False)] = False
        else:
            values[int(self.rng.integers(1, n - 1))] = False
        self.walls[:, col] = values
        self._features(col=col)

    def _features(self, row: int | None = None, col: int | None = None) -> None:
        key = (row, slice(None)) if row is not None else (slice(None), col)
        floor, size = ~self.walls[key], self.walls[key].size
        self.orbs[key] = floor & (self.rng.random(size) < self.cfg.orb_density) if self.level >= 1 else False
        self.spikes[key] = floor & (self.rng.random(size) < self.cfg.spike_density) if self.level >= 2 else False
        self.spawns[key] = floor & (self.rng.random(size) < self.cfg.enemy_spawn_density) if self.level >= 3 else False
