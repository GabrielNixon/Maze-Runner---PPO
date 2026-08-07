from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mazerunner_ppo.rendering import make_frame

_GRID_SHAPE = (7, 19, 19)
_STATS_SHAPE = (14,)
_EVENT_ORB = 1
_EVENT_SPIKE = 2
_EVENT_BLOCKED = 4
_ACTIVE_ENV = False


class ExactBackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ExactGameConfig:
    frame_skip: int = 6
    max_episode_seconds: float = 600.0
    survival_reward: float = 0.02
    orb_reward: float = 2.0
    spike_penalty: float = -0.5
    death_penalty: float = -2.0
    blocked_move_penalty: float = -0.005
    enemy_distance_scale: float = 0.08
    hungry_orb_distance_scale: float = 0.06

    @property
    def decision_dt(self) -> float:
        return self.frame_skip / 60.0

    @property
    def max_episode_steps(self) -> int:
        return int(self.max_episode_seconds / self.decision_dt)

    def validate(self) -> None:
        if not 1 <= self.frame_skip <= 120:
            raise ValueError("frame_skip must be between 1 and 120")
        if self.max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive")


def _library_filename() -> str:
    if platform.system() == "Darwin":
        return "libmazerunner_exact.dylib"
    if platform.system() == "Linux":
        return "libmazerunner_exact.so"
    raise ExactBackendUnavailable("The exact backend currently supports macOS and Linux")


def find_exact_library() -> Path:
    override = os.environ.get("MAZERUNNER_EXACT_LIB")
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(Path(__file__).resolve().parent / "lib" / _library_filename())
    candidates.append(Path.cwd() / "mazerunner_ppo" / "lib" / _library_filename())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    command = "python scripts/bootstrap_exact.py"
    raise ExactBackendUnavailable(
        "Exact MazeRunner library not found. Build it first with: " + command
    )


def _configure_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    float_ptr = ctypes.POINTER(ctypes.c_float)
    lib.mr_set_curriculum.argtypes = [ctypes.c_int]
    lib.mr_set_curriculum.restype = None
    lib.mr_reset.argtypes = [ctypes.c_uint32]
    lib.mr_reset.restype = ctypes.c_int
    lib.mr_step.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.mr_step.restype = ctypes.c_int
    lib.mr_get_observation.argtypes = [float_ptr, float_ptr]
    lib.mr_get_observation.restype = None
    lib.mr_get_events.argtypes = []
    lib.mr_get_events.restype = ctypes.c_int
    lib.mr_is_done.argtypes = []
    lib.mr_is_done.restype = ctypes.c_int
    lib.mr_get_death_reason.argtypes = []
    lib.mr_get_death_reason.restype = ctypes.c_int
    lib.mr_get_survival_time.argtypes = []
    lib.mr_get_survival_time.restype = ctypes.c_float
    lib.mr_get_hunger.argtypes = []
    lib.mr_get_hunger.restype = ctypes.c_float
    lib.mr_get_orbs.argtypes = []
    lib.mr_get_orbs.restype = ctypes.c_int
    lib.mr_get_enemy_count.argtypes = []
    lib.mr_get_enemy_count.restype = ctypes.c_int
    lib.mr_get_spike_phase.argtypes = []
    lib.mr_get_spike_phase.restype = ctypes.c_float
    lib.mr_upstream_abi_version.argtypes = []
    lib.mr_upstream_abi_version.restype = ctypes.c_uint32
    if lib.mr_upstream_abi_version() != 1:
        raise ExactBackendUnavailable("Unsupported exact backend ABI")
    return lib


def _nearest_distance(channel: np.ndarray) -> float:
    positions = np.argwhere(channel > 0.5)
    if positions.size == 0:
        return 1.0
    center = np.array([channel.shape[0] // 2, channel.shape[1] // 2])
    distances = np.max(np.abs(positions - center), axis=1)
    return float(min(1.0, distances.min() / max(center[0], 1)))


class ExactMazeRunnerEnv(gym.Env[dict[str, np.ndarray], int]):
    """Gymnasium wrapper around the real MazeRunner C simulation.

    The C bridge uses process-global state because the upstream game does the same.
    Run one environment per process; `scripts/train_exact.py` enforces that with
    `SubprocVecEnv`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: ExactGameConfig | None = None,
        render_mode: str | None = None,
        curriculum_level: int = 3,
        library_path: str | Path | None = None,
    ) -> None:
        global _ACTIVE_ENV
        if _ACTIVE_ENV:
            raise RuntimeError(
                "Only one ExactMazeRunnerEnv may exist per process; use SubprocVecEnv"
            )
        super().__init__()
        self.config = config or ExactGameConfig()
        self.config.validate()
        if curriculum_level not in range(4):
            raise ValueError("curriculum_level must be 0..3")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("invalid render_mode")
        path = Path(library_path).expanduser().resolve() if library_path else find_exact_library()
        self._lib = _configure_library(path)
        self.library_path = path
        self.curriculum_level = curriculum_level
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, _GRID_SHAPE, np.float32),
                "stats": spaces.Box(0.0, 1.0, _STATS_SHAPE, np.float32),
            }
        )
        self._grid = np.zeros(_GRID_SHAPE, dtype=np.float32)
        self._stats = np.zeros(_STATS_SHAPE, dtype=np.float32)
        self._steps = 0
        self._last_orbs = 0
        self._previous_enemy_distance = 1.0
        self._previous_orb_distance = 1.0
        self._pygame = self._window = self._clock = None
        _ACTIVE_ENV = True

    def _read_observation(self) -> dict[str, np.ndarray]:
        self._lib.mr_get_observation(
            self._grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._stats.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return {"grid": self._grid.copy(), "stats": self._stats.copy()}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        if seed is None:
            seed = int(self.np_random.integers(0, 2**32 - 1))
        self._lib.mr_set_curriculum(self.curriculum_level)
        if self._lib.mr_reset(ctypes.c_uint32(seed)) != 1:
            raise RuntimeError("Exact backend reset failed")
        self._steps = 0
        self._last_orbs = 0
        observation = self._read_observation()
        self._previous_enemy_distance = _nearest_distance(observation["grid"][5] + observation["grid"][6])
        self._previous_orb_distance = _nearest_distance(observation["grid"][1])
        if self.render_mode == "human":
            self.render()
        return observation, self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        result = self._lib.mr_step(int(action), self.config.frame_skip)
        if result < 0:
            raise RuntimeError("Exact backend rejected the action")
        self._steps += 1
        observation = self._read_observation()
        events = int(self._lib.mr_get_events())
        reward = self.config.survival_reward
        if events & _EVENT_ORB:
            reward += self.config.orb_reward
        if events & _EVENT_SPIKE:
            reward += self.config.spike_penalty
        if events & _EVENT_BLOCKED:
            reward += self.config.blocked_move_penalty

        enemy_distance = _nearest_distance(observation["grid"][5] + observation["grid"][6])
        reward += self.config.enemy_distance_scale * (
            enemy_distance - self._previous_enemy_distance
        )
        self._previous_enemy_distance = enemy_distance

        orb_distance = _nearest_distance(observation["grid"][1])
        hunger = float(self._lib.mr_get_hunger())
        if hunger < 0.65:
            reward += self.config.hungry_orb_distance_scale * (
                self._previous_orb_distance - orb_distance
            )
        self._previous_orb_distance = orb_distance

        terminated = bool(self._lib.mr_is_done())
        if terminated:
            reward += self.config.death_penalty
        truncated = not terminated and self._steps >= self.config.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        reason = {0: None, 1: "enemy", 2: "hunger"}.get(
            int(self._lib.mr_get_death_reason()), "unknown"
        )
        return {
            "survival_steps": self._steps,
            "survival_seconds": float(self._lib.mr_get_survival_time()),
            "orbs_collected": int(self._lib.mr_get_orbs()),
            "hunger": float(self._lib.mr_get_hunger()),
            "enemy_count": int(self._lib.mr_get_enemy_count()),
            "spike_phase": float(self._lib.mr_get_spike_phase()),
            "death_reason": reason,
            "curriculum_level": self.curriculum_level,
            "backend": "exact-c",
        }

    def render(self):
        frame = make_frame(self._grid, float(self._lib.mr_get_hunger()))
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
        self._clock.tick(max(1, round(60 / self.config.frame_skip)))
        return None

    def close(self) -> None:
        global _ACTIVE_ENV
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = self._window = self._clock = None
        _ACTIVE_ENV = False
