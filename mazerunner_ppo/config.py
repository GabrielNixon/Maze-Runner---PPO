from __future__ import annotations

from dataclasses import dataclass

# stay, N, NE, E, SE, S, SW, W, NW
ACTION_DELTAS = (
    (0, 0), (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
)


@dataclass(frozen=True)
class MazeRunnerConfig:
    buffer_size: int = 31
    observation_size: int = 19
    decision_dt: float = 0.20
    hunger_decay_per_second: float = 0.10
    orb_restore: float = 0.50
    spike_damage: float = 0.50
    spike_period_steps: int = 15
    spike_warning_start: int = 8
    spike_up_start: int = 10
    spike_damage_cooldown_steps: int = 7
    enemy_move_interval: int = 2
    enemy_freeze_steps: int = 5
    max_enemies: int = 24
    max_episode_steps: int = 2_000
    orb_density: float = 0.055
    spike_density: float = 0.030
    enemy_spawn_density: float = 0.014
    survival_reward: float = 0.01
    orb_reward: float = 1.0
    spike_penalty: float = -0.25
    death_penalty: float = -10.0
    blocked_move_penalty: float = -0.002

    def validate(self) -> None:
        if self.buffer_size < 15 or self.buffer_size % 2 == 0:
            raise ValueError("buffer_size must be odd and >= 15")
        if self.observation_size < 7 or self.observation_size % 2 == 0:
            raise ValueError("observation_size must be odd and >= 7")
        if self.observation_size > self.buffer_size:
            raise ValueError("observation_size cannot exceed buffer_size")
        if not 0 < self.spike_warning_start < self.spike_up_start < self.spike_period_steps:
            raise ValueError("invalid spike timing")
