from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

from mazerunner_ppo.env import MazeRunnerEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained recurrent PPO agent")
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1_000)
    parser.add_argument("--curriculum-level", type=int, choices=range(4), default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "human" if args.render else None
    env = MazeRunnerEnv(render_mode=render_mode, curriculum_level=args.curriculum_level)
    model = RecurrentPPO.load(args.model)

    records: list[dict[str, object]] = []
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        terminated = truncated = False

        while not (terminated or truncated):
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=not args.stochastic,
            )
            obs, _, terminated, truncated, info = env.step(int(action))
            episode_start[:] = terminated or truncated

        records.append(
            {
                "episode": episode,
                "seed": args.seed + episode,
                "survival_seconds": info["survival_seconds"],
                "orbs_collected": info["orbs_collected"],
                "death_reason": info["death_reason"],
            }
        )

    env.close()
    times = np.array([float(r["survival_seconds"]) for r in records])
    orbs = np.array([int(r["orbs_collected"]) for r in records])
    summary = {
        "episodes": args.episodes,
        "mean_survival_seconds": float(times.mean()),
        "median_survival_seconds": float(np.median(times)),
        "max_survival_seconds": float(times.max()),
        "mean_orbs": float(orbs.mean()),
        "episodes_detail": records,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
