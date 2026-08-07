from gymnasium.envs.registration import register

from mazerunner_ppo.env import MazeRunnerConfig, MazeRunnerEnv

register(
    id="MazeRunnerPPO-v0",
    entry_point="mazerunner_ppo.env:MazeRunnerEnv",
)

__all__ = ["MazeRunnerConfig", "MazeRunnerEnv"]
