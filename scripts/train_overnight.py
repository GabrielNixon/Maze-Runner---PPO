from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from mazerunner_ppo.exact_env import ExactGameConfig, ExactMazeRunnerEnv
from mazerunner_ppo.foraging import ForagingRewardConfig, ForagingRewardWrapper
from mazerunner_ppo.record import SELECTION_METRICS, RecordEvalCallback

OVERNIGHT_MILESTONE_REWARDS = (
    (60.0, 8.0),
    (120.0, 16.0),
    (180.0, 28.0),
    (240.0, 42.0),
    (265.0, 60.0),
    (300.0, 75.0),
    (360.0, 95.0),
    (420.0, 120.0),
)
OVERNIGHT_MILESTONES = tuple(seconds for seconds, _ in OVERNIGHT_MILESTONE_REWARDS)


@dataclass(frozen=True)
class Phase:
    name: str
    fraction: float
    curriculum_ratios: tuple[float, float, float]  # levels 1, 2, 3
    learning_rate: float
    ent_coef: float
    shaping_scale: float


PHASES = (
    Phase(
        name="skill_school",
        fraction=0.25,
        curriculum_ratios=(0.375, 0.375, 0.25),
        learning_rate=4e-5,
        ent_coef=0.006,
        shaping_scale=1.0,
    ),
    Phase(
        name="integration",
        fraction=0.50,
        curriculum_ratios=(0.125, 0.25, 0.625),
        learning_rate=2.5e-5,
        ent_coef=0.004,
        shaping_scale=0.75,
    ),
    Phase(
        name="record_polish",
        fraction=0.25,
        curriculum_ratios=(0.0, 0.125, 0.875),
        learning_rate=1e-5,
        ent_coef=0.002,
        shaping_scale=0.45,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overnight multi-stage fine-tuning for food routing, spike awareness, "
            "and full-game survival"
        )
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("runs/record-chase/final_record_model.zip"),
        help="Strong full-game checkpoint to continue from",
    )
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--eval-seed", type=int, default=40_000)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=200_000)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="p90")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--output", type=Path, default=Path("runs/overnight-skills"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def game_config(frame_skip: int) -> ExactGameConfig:
    return ExactGameConfig(
        frame_skip=frame_skip,
        max_episode_seconds=480.0,
        survival_reward=0.03,
        # Raw pickup reward stays small. Most food reward below depends on how
        # much of the +0.5 refill is actually useful, discouraging waste at full hunger.
        orb_reward=0.35,
        # A real spike hit removes 0.5 hunger, so it deserves a much stronger
        # signal than the old -0.5 reward.
        spike_penalty=-2.0,
        death_penalty=-6.0,
        blocked_move_penalty=-0.005,
        enemy_distance_scale=0.08,
        # Replaced by wall- and spike-aware shortest-path shaping in the wrapper.
        hungry_orb_distance_scale=0.0,
        hunger_orb_shape_start=0.75,
        hunger_orb_urgency_power=2.0,
        survival_milestone_rewards=OVERNIGHT_MILESTONE_REWARDS,
    )


def allocate_curricula(num_envs: int, ratios: tuple[float, float, float]) -> list[int]:
    """Largest-remainder allocation of envs to curriculum levels 1/2/3."""

    raw = [num_envs * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = num_envs - sum(counts)
    order = sorted(range(3), key=lambda index: raw[index] - counts[index], reverse=True)
    for index in order[:remaining]:
        counts[index] += 1

    # Every phase must preserve at least one full-game environment.
    if counts[2] == 0:
        donor = 0 if counts[0] >= counts[1] else 1
        counts[donor] -= 1
        counts[2] = 1

    return [1] * counts[0] + [2] * counts[1] + [3] * counts[2]


def skill_reward_config(curriculum_level: int, shaping_scale: float) -> ForagingRewardConfig:
    # Clean Level-1 food episodes receive the strongest routing signal. Level 2
    # gets extra spike teaching without enemies. Full Level 3 uses gentler
    # shaping so the already-good enemy behavior is not overwritten.
    food_multiplier = {1: 1.25, 2: 1.05, 3: 0.80}[curriculum_level]
    hazard_multiplier = {1: 0.0, 2: 1.25, 3: 1.0}[curriculum_level]
    hazard_scale = max(0.70, shaping_scale)
    return ForagingRewardConfig(
        path_scale=0.60 * shaping_scale * food_multiplier,
        idle_path_fraction=0.15,
        useful_orb_bonus=4.0 * shaping_scale * food_multiplier,
        action_spike_penalty=1.5 * hazard_scale * hazard_multiplier,
        center_spike_penalty=0.55 * hazard_scale * hazard_multiplier,
        spike_hit_penalty=3.5 * hazard_scale * hazard_multiplier,
        safe_spike_cost=0.15,
        warning_spike_cost=4.0,
        active_spike_cost=10.0,
    )


def make_train_env(
    rank: int,
    seed: int,
    frame_skip: int,
    curriculum_level: int,
    shaping_scale: float,
):
    def factory():
        env = ExactMazeRunnerEnv(
            config=game_config(frame_skip),
            curriculum_level=curriculum_level,
        )
        env = ForagingRewardWrapper(
            env,
            skill_reward_config(curriculum_level, shaping_scale),
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    return factory


def make_eval_env(seed: int, frame_skip: int):
    def factory():
        env = ExactMazeRunnerEnv(
            config=game_config(frame_skip),
            curriculum_level=3,
        )
        # Zero reward coefficients: the wrapper is present only to measure food
        # efficiency and spike mistakes on the held-out full-game episodes.
        env = ForagingRewardWrapper(
            env,
            ForagingRewardConfig(
                path_scale=0.0,
                useful_orb_bonus=0.0,
                action_spike_penalty=0.0,
                center_spike_penalty=0.0,
                spike_hit_penalty=0.0,
                safe_spike_cost=0.15,
                warning_spike_cost=4.0,
                active_spike_cost=10.0,
            ),
        )
        env.reset(seed=seed)
        return Monitor(env)

    return factory


def make_train_vec(
    phase: Phase,
    *,
    num_envs: int,
    seed: int,
    frame_skip: int,
    phase_index: int,
) -> tuple[VecMonitor, list[int]]:
    levels = allocate_curricula(num_envs, phase.curriculum_ratios)
    phase_seed = seed + phase_index * 100_000
    env = VecMonitor(
        SubprocVecEnv(
            [
                make_train_env(
                    rank,
                    phase_seed,
                    frame_skip,
                    level,
                    phase.shaping_scale,
                )
                for rank, level in enumerate(levels)
            ],
            start_method="spawn",
        )
    )
    return env, levels


def set_optimizer_phase(model: RecurrentPPO, phase: Phase) -> None:
    learning_rate = phase.learning_rate
    model.learning_rate = learning_rate
    model.lr_schedule = lambda _: learning_rate
    model.ent_coef = phase.ent_coef
    for group in model.policy.optimizer.param_groups:
        group["lr"] = learning_rate


def phase_timesteps(total_timesteps: int) -> list[int]:
    steps = [int(total_timesteps * phase.fraction) for phase in PHASES]
    steps[-1] += total_timesteps - sum(steps)
    return steps


def main() -> None:
    args = parse_args()
    if args.num_envs < 4:
        raise ValueError("--num-envs must be at least four for the mixed curriculum")
    if args.timesteps < args.num_envs:
        raise ValueError("--timesteps is too small")
    if args.eval_episodes < 1 or args.eval_every < 1:
        raise ValueError("evaluation settings must be positive")
    if not args.resume.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    best_dir = args.output / "best"
    checkpoint_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)

    eval_env = SubprocVecEnv(
        [make_eval_env(args.eval_seed, args.frame_skip)],
        start_method="spawn",
    )
    record_callback = RecordEvalCallback(
        eval_env,
        eval_freq=max(args.eval_every // args.num_envs, 1),
        n_eval_episodes=args.eval_episodes,
        seed=args.eval_seed,
        save_path=best_dir,
        log_path=args.output / "overnight_eval.jsonl",
        selection_metric=args.selection_metric,
        milestones=OVERNIGHT_MILESTONES,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500_000 // args.num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="overnight_recurrent_ppo",
    )

    model: RecurrentPPO | None = None
    train_env: VecMonitor | None = None
    steps_by_phase = phase_timesteps(args.timesteps)

    print(
        "Overnight survival-skills run | "
        f"resume={args.resume} total={args.timesteps:,} envs={args.num_envs} "
        f"selection={args.selection_metric}"
    )

    try:
        for phase_index, (phase, phase_steps) in enumerate(zip(PHASES, steps_by_phase, strict=True)):
            new_env, levels = make_train_vec(
                phase,
                num_envs=args.num_envs,
                seed=args.seed,
                frame_skip=args.frame_skip,
                phase_index=phase_index,
            )
            if model is None:
                custom_objects = {
                    "learning_rate": phase.learning_rate,
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "ent_coef": phase.ent_coef,
                }
                model = RecurrentPPO.load(
                    args.resume,
                    env=new_env,
                    device=args.device,
                    custom_objects=custom_objects,
                )
                model.tensorboard_log = str(args.output / "tensorboard")
            else:
                old_env = train_env
                model.set_env(new_env)
                if old_env is not None:
                    old_env.close()

            train_env = new_env
            set_optimizer_phase(model, phase)
            counts = {level: levels.count(level) for level in (1, 2, 3)}
            print(
                f"\n=== Phase {phase_index + 1}/{len(PHASES)}: {phase.name} ===\n"
                f"steps={phase_steps:,} lr={phase.learning_rate:g} ent={phase.ent_coef:g} "
                f"shaping={phase.shaping_scale:g}\n"
                f"curriculum envs: L1 food={counts[1]} | L2 food+spikes={counts[2]} | "
                f"L3 full={counts[3]}"
            )

            # Re-evaluate at each phase boundary. The callback retains best_score,
            # so a later specialist cannot overwrite a stronger full-game model.
            record_callback.evaluate_at_start = True
            model.learn(
                total_timesteps=phase_steps,
                callback=[checkpoint_callback, record_callback],
                progress_bar=True,
                reset_num_timesteps=False,
            )
            model.save(args.output / f"after_{phase_index + 1}_{phase.name}")

        assert model is not None
        model.save(args.output / "final_overnight_model")
        print(
            "\nOvernight training complete. Compare:\n"
            f"  {best_dir / 'best_record_model.zip'}\n"
            f"  {args.output / 'final_overnight_model.zip'}"
        )
    finally:
        if train_env is not None:
            train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
