from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from mazerunner_ppo.exact_env import ExactGameConfig, ExactMazeRunnerEnv
from mazerunner_ppo.record import RecordEvalCallback, SELECTION_METRICS

RECORD_MILESTONE_REWARDS = (
    (60.0, 10.0),
    (120.0, 20.0),
    (180.0, 30.0),
    (240.0, 40.0),
    (300.0, 60.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an exact-game recurrent PPO policy for a 5-minute record run"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("runs/exact-full/best/best_model.zip"),
        help="Full-game checkpoint to fine-tune",
    )
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="p90")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--output", type=Path, default=Path("runs/record-chase"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def record_config(frame_skip: int) -> ExactGameConfig:
    return ExactGameConfig(
        frame_skip=frame_skip,
        max_episode_seconds=360.0,
        survival_reward=0.03,
        orb_reward=1.5,
        spike_penalty=-0.5,
        death_penalty=-5.0,
        blocked_move_penalty=-0.005,
        enemy_distance_scale=0.08,
        hungry_orb_distance_scale=0.08,
        hunger_orb_shape_start=0.75,
        hunger_orb_urgency_power=2.0,
        survival_milestone_rewards=RECORD_MILESTONE_REWARDS,
    )


def make_env(rank: int, seed: int, frame_skip: int):
    def factory():
        env = ExactMazeRunnerEnv(
            config=record_config(frame_skip),
            curriculum_level=3,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    return factory


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least one")
    if args.eval_episodes < 1 or args.eval_every < 1:
        raise ValueError("evaluation settings must be positive")
    if not args.resume.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    best_dir = args.output / "best"
    checkpoint_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)

    train_env = VecMonitor(
        SubprocVecEnv(
            [make_env(rank, args.seed, args.frame_skip) for rank in range(args.num_envs)],
            start_method="spawn",
        )
    )
    eval_env = SubprocVecEnv(
        [make_env(0, args.eval_seed, args.frame_skip)],
        start_method="spawn",
    )

    custom_objects = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
    }
    model = RecurrentPPO.load(
        args.resume,
        env=train_env,
        device=args.device,
        custom_objects=custom_objects,
    )
    model.tensorboard_log = str(args.output / "tensorboard")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(250_000 // args.num_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="record_recurrent_ppo",
    )
    record_callback = RecordEvalCallback(
        eval_env,
        eval_freq=max(args.eval_every // args.num_envs, 1),
        n_eval_episodes=args.eval_episodes,
        seed=args.eval_seed,
        save_path=best_dir,
        log_path=args.output / "record_eval.jsonl",
        selection_metric=args.selection_metric,
        verbose=1,
    )

    print(
        "Record chase starting from "
        f"{args.resume} | lr={args.learning_rate:g} gamma={args.gamma:g} "
        f"selection={args.selection_metric}"
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, record_callback],
            progress_bar=True,
            reset_num_timesteps=False,
        )
        model.save(args.output / "final_record_model")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
