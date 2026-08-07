from gymnasium.envs.registration import register

from mazerunner_ppo.env import MazeRunnerConfig, MazeRunnerEnv
from mazerunner_ppo.exact_env import ExactGameConfig, ExactMazeRunnerEnv

register(
    id="MazeRunnerPPO-v0",
    entry_point="mazerunner_ppo.env:MazeRunnerEnv",
)
register(
    id="MazeRunnerExact-v0",
    entry_point="mazerunner_ppo.exact_env:ExactMazeRunnerEnv",
)

__all__ = [
    "ExactGameConfig",
    "ExactMazeRunnerEnv",
    "MazeRunnerConfig",
    "MazeRunnerEnv",
]
