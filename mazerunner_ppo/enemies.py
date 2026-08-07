from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from mazerunner_ppo.config import ACTION_DELTAS, MazeRunnerConfig
from mazerunner_ppo.generation import RollingMap


@dataclass
class Enemy:
    row: int
    col: int
    frozen: int


class EnemyManager:
    def __init__(
        self,
        cfg: MazeRunnerConfig,
        rng: np.random.Generator,
        world: RollingMap,
        level: int,
    ) -> None:
        self.cfg, self.rng, self.world, self.level = cfg, rng, world, level
        self.items: list[Enemy] = []

    def shift(self, dr: int, dc: int) -> None:
        n = self.cfg.buffer_size
        for enemy in self.items:
            enemy.row += dr
            enemy.col += dc
        self.items[:] = [e for e in self.items if 0 <= e.row < n and 0 <= e.col < n]

    def drain_spawns(self) -> None:
        if self.level < 3:
            return
        cr, cc = self.world.center
        radius = self.cfg.observation_size // 2
        positions = list(zip(*np.where(self.world.spawns), strict=False))
        self.rng.shuffle(positions)
        for row, col in positions:
            if len(self.items) >= self.cfg.max_enemies:
                break
            if max(abs(row - cr), abs(col - cc)) <= radius:
                self.world.spawns[row, col] = False
                self.items.append(Enemy(int(row), int(col), self.cfg.enemy_freeze_steps))

    def update(self, step: int, spikes_up: bool) -> None:
        if self.level < 3:
            self.items.clear()
            return
        move = step % self.cfg.enemy_move_interval == 0
        occupied: set[tuple[int, int]] = set()
        alive = []
        for enemy in self.items:
            if enemy.frozen:
                enemy.frozen -= 1
            elif move:
                nxt = self._bfs((enemy.row, enemy.col), self.world.center)
                if nxt and nxt not in occupied:
                    enemy.row, enemy.col = nxt
            if spikes_up and self.world.spikes[enemy.row, enemy.col]:
                continue
            occupied.add((enemy.row, enemy.col))
            alive.append(enemy)
        self.items = alive

    def _bfs(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[int, int] | None:
        if start == goal:
            return start
        n = self.cfg.buffer_size
        queue = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while queue:
            row, col = queue.popleft()
            for dx, dy in ACTION_DELTAS[1:]:
                nxt = nr, nc = row + dy, col + dx
                if not (0 <= nr < n and 0 <= nc < n) or nxt in parent or self.world.walls[nxt]:
                    continue
                if dx and dy and (
                    self.world.walls[row, col + dx]
                    or self.world.walls[row + dy, col]
                ):
                    continue
                parent[nxt] = (row, col)
                if nxt == goal:
                    queue.clear()
                    break
                queue.append(nxt)
        if goal not in parent:
            return None
        node = goal
        while parent[node] not in (None, start):
            node = parent[node]  # type: ignore[assignment]
        return node
