from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

DEFAULT_RECORD_MILESTONES = (60.0, 120.0, 180.0, 240.0, 300.0)
SELECTION_METRICS = ("mean", "median", "p90", "p95", "max")


def summarize_record_runs(
    survival_seconds: Sequence[float],
    orbs_collected: Sequence[float] | None = None,
    death_reasons: Sequence[str | None] | None = None,
    milestones: Sequence[float] = DEFAULT_RECORD_MILESTONES,
) -> dict[str, Any]:
    survival = np.asarray(survival_seconds, dtype=float)
    if survival.size == 0:
        raise ValueError("at least one episode is required")

    metrics: dict[str, Any] = {
        "episodes": int(survival.size),
        "mean": float(survival.mean()),
        "median": float(np.median(survival)),
        "p90": float(np.percentile(survival, 90)),
        "p95": float(np.percentile(survival, 95)),
        "max": float(survival.max()),
    }
    if orbs_collected is not None:
        orbs = np.asarray(orbs_collected, dtype=float)
        if orbs.size != survival.size:
            raise ValueError("orbs_collected must match survival_seconds length")
        metrics["mean_orbs"] = float(orbs.mean())

    metrics["milestone_rates"] = {
        f"over_{int(threshold)}s": float(np.mean(survival >= threshold))
        for threshold in milestones
    }

    if death_reasons is not None:
        if len(death_reasons) != survival.size:
            raise ValueError("death_reasons must match survival_seconds length")
        counts = Counter(reason or "truncated" for reason in death_reasons)
        metrics["death_counts"] = dict(sorted(counts.items()))

    return metrics


def _vectorize_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.expand_dims(value, axis=0) for key, value in observation.items()}


class RecordEvalCallback(BaseCallback):
    """Evaluate recurrent PPO on fixed seeds and save the best survival-tail model."""

    def __init__(
        self,
        eval_env: VecEnv,
        *,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int,
        save_path: Path,
        log_path: Path,
        selection_metric: str = "p90",
        milestones: Sequence[float] = DEFAULT_RECORD_MILESTONES,
        deterministic: bool = True,
        evaluate_at_start: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        if eval_freq < 1:
            raise ValueError("eval_freq must be positive")
        if n_eval_episodes < 1:
            raise ValueError("n_eval_episodes must be positive")
        if selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
        if eval_env.num_envs != 1:
            raise ValueError("RecordEvalCallback requires exactly one evaluation environment")

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.save_path = Path(save_path)
        self.log_path = Path(log_path)
        self.selection_metric = selection_metric
        self.milestones = tuple(float(value) for value in milestones)
        self.deterministic = deterministic
        self.evaluate_at_start = evaluate_at_start
        self.best_score = -np.inf

    def _on_training_start(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.evaluate_at_start:
            self._run_evaluation()

    def _reset_with_seed(self, seed: int) -> dict[str, np.ndarray]:
        result = self.eval_env.env_method("reset", seed=seed)[0]
        observation, _ = result
        return _vectorize_observation(observation)

    def _run_episode(self, seed: int) -> tuple[float, int, str | None]:
        observation = self._reset_with_seed(seed)
        state = None
        episode_start = np.ones((1,), dtype=bool)
        done = np.zeros((1,), dtype=bool)
        info: dict[str, Any] = {}

        while not bool(done[0]):
            action, state = self.model.predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=self.deterministic,
            )
            observation, _, done, infos = self.eval_env.step(action)
            episode_start = done
            info = infos[0]

        return (
            float(info["survival_seconds"]),
            int(info["orbs_collected"]),
            info.get("death_reason"),
        )

    def _run_evaluation(self) -> dict[str, Any]:
        survival: list[float] = []
        orbs: list[int] = []
        reasons: list[str | None] = []
        for offset in range(self.n_eval_episodes):
            seconds, orb_count, reason = self._run_episode(self.seed + offset)
            survival.append(seconds)
            orbs.append(orb_count)
            reasons.append(reason)

        metrics = summarize_record_runs(
            survival,
            orbs,
            reasons,
            milestones=self.milestones,
        )
        metrics["timesteps"] = int(self.num_timesteps)
        metrics["selection_metric"] = self.selection_metric
        metrics["selection_score"] = float(metrics[self.selection_metric])

        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

        for key in ("mean", "median", "p90", "p95", "max", "mean_orbs"):
            self.logger.record(f"record/{key}", metrics[key])
        for label, rate in metrics["milestone_rates"].items():
            self.logger.record(f"record/{label}", rate)
        self.logger.record("record/selection_score", metrics["selection_score"])
        self.logger.dump(self.num_timesteps)

        score = metrics["selection_score"]
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.save_path / "best_record_model")
            with (self.save_path / "best_record_metrics.json").open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(metrics, handle, indent=2, sort_keys=True)
            if self.verbose:
                print(
                    f"New best record model: {self.selection_metric}={score:.2f}s "
                    f"at {self.num_timesteps:,} timesteps"
                )

        if self.verbose:
            rates = metrics["milestone_rates"]
            print(
                "Record eval: "
                f"median={metrics['median']:.1f}s "
                f"p90={metrics['p90']:.1f}s "
                f"p95={metrics['p95']:.1f}s "
                f"max={metrics['max']:.1f}s "
                f">180s={rates['over_180s']:.0%} "
                f">300s={rates['over_300s']:.0%}"
            )
        return metrics

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
        return True
