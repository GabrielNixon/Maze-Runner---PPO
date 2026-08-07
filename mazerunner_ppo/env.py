from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mazerunner_ppo.config import ACTION_DELTAS, MazeRunnerConfig
from mazerunner_ppo.enemies import EnemyManager
from mazerunner_ppo.generation import RollingMap
from mazerunner_ppo.observations import build_info, build_observation
from mazerunner_ppo.rendering import make_frame


class MazeRunnerEnv(gym.Env[dict[str, np.ndarray], int]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        config: MazeRunnerConfig | None = None,
        render_mode: str | None = None,
        curriculum_level: int = 3,
    ) -> None:
        super().__init__()
        self.config = config or MazeRunnerConfig()
        self.config.validate()
        if curriculum_level not in range(4):
            raise ValueError("curriculum_level must be 0..3")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("invalid render_mode")
        self.render_mode, self.curriculum_level = render_mode, curriculum_level
        n = self.config.observation_size
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (7, n, n), np.float32),
            "stats": spaces.Box(0.0, 1.0, (14,), np.float32),
        })
        self.rng = np.random.default_rng()
        self.world = RollingMap(self.config, self.rng, curriculum_level)
        self.enemies = EnemyManager(self.config, self.rng, self.world, curriculum_level)
        self.hunger = 1.0
        self.steps = self.orbs = self.last_action = 0
        self.last_spike_hit = -10_000
        self.death_reason: str | None = None
        self._pygame = self._window = self._clock = None

    @property
    def spike_phase(self) -> int:
        return self.steps % self.config.spike_period_steps

    @property
    def spikes_up(self) -> bool:
        return self.spike_phase >= self.config.spike_up_start

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.world = RollingMap(self.config, self.rng, self.curriculum_level)
        self.world.reset()
        self.enemies = EnemyManager(self.config, self.rng, self.world, self.curriculum_level)
        self.enemies.drain_spawns()
        self.hunger = 1.0
        self.steps = self.orbs = self.last_action = 0
        self.last_spike_hit = -10_000
        self.death_reason = None
        if self.render_mode == "human":
            self.render()
        return self._observation(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        self.steps += 1
        self.last_action = int(action)
        reward = self.config.survival_reward
        dx, dy = ACTION_DELTAS[action]
        if self.world.can_move(dx, dy):
            dr, dc = self.world.shift(dx, dy)
            self.enemies.shift(dr, dc)
        elif action:
            reward += self.config.blocked_move_penalty

        self.enemies.drain_spawns()
        self.enemies.update(self.steps, self.spikes_up)
        if self.curriculum_level >= 1:
            self.hunger = max(
                0.0,
                self.hunger - self.config.hunger_decay_per_second * self.config.decision_dt,
            )
        cr, cc = self.world.center
        if self.curriculum_level >= 1 and self.world.orbs[cr, cc]:
            self.world.orbs[cr, cc] = False
            self.hunger = min(1.0, self.hunger + self.config.orb_restore)
            self.orbs += 1
            reward += self.config.orb_reward
        if self.curriculum_level >= 2 and self.spikes_up and self.world.spikes[cr, cc]:
            if self.steps - self.last_spike_hit > self.config.spike_damage_cooldown_steps:
                self.hunger = max(0.0, self.hunger - self.config.spike_damage)
                self.last_spike_hit = self.steps
                reward += self.config.spike_penalty

        enemy_hit = any((e.row, e.col) == (cr, cc) for e in self.enemies.items)
        terminated = enemy_hit or (self.curriculum_level >= 1 and self.hunger <= 0)
        if terminated:
            self.death_reason = "enemy" if enemy_hit else "hunger"
            reward += self.config.death_penalty
        truncated = not terminated and self.steps >= self.config.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), terminated, truncated, self._info()

    def _observation(self) -> dict[str, np.ndarray]:
        return build_observation(self)

    def _info(self) -> dict[str, Any]:
        return build_info(self)

    def render(self):
        frame = make_frame(self._observation()["grid"], self.hunger)
        if self.render_mode == "rgb_array":
            return frame
        if self.render_mode != "human":
            return None
        import pygame
        if self._pygame is None:
            self._pygame = pygame
            pygame.init()
            self._window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            self._clock = pygame.time.Clock()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._window.blit(surface, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = self._window = self._clock = None
