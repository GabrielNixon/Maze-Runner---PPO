from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

from mazerunner_ppo.exact_env import ExactGameConfig, ExactMazeRunnerEnv
from mazerunner_ppo.record import DEFAULT_RECORD_MILESTONES, summarize_record_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a policy on the exact C game")
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--curriculum-level", type=int, choices=range(4), default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = ExactMazeRunnerEnv(
        config=ExactGameConfig(frame_skip=args.frame_skip),
        curriculum_level=args.curriculum_level,
        render_mode="human" if args.render else None,
    )
    model = RecurrentPPO.load(args.model)
    details = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            state = None
            episode_start = np.ones((1,), dtype=bool)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action, state = model.predict(
                    obs,
                    state=state,
                    episode_start=episode_start,
                    deterministic=not args.stochastic,
                )
                obs, _, terminated, truncated, info = env.step(int(action))
                episode_start[:] = terminated or truncated
            details.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "survival_seconds": info["survival_seconds"],
                    "orbs_collected": info["orbs_collected"],
                    "death_reason": info["death_reason"],
                }
            )
    finally:
        env.close()

    survival = [row["survival_seconds"] for row in details]
    orbs = [row["orbs_collected"] for row in details]
    reasons = [row["death_reason"] for row in details]
    record_metrics = summarize_record_runs(
        survival,
        orbs,
        reasons,
        milestones=DEFAULT_RECORD_MILESTONES,
    )
    result = {
        "backend": "exact-c",
        "episodes": args.episodes,
        "mean_survival_seconds": record_metrics["mean"],
        "median_survival_seconds": record_metrics["median"],
        "p90_survival_seconds": record_metrics["p90"],
        "p95_survival_seconds": record_metrics["p95"],
        "max_survival_seconds": record_metrics["max"],
        "mean_orbs": record_metrics["mean_orbs"],
        "milestone_rates": record_metrics["milestone_rates"],
        "death_counts": record_metrics["death_counts"],
        "episodes_detail": details,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
