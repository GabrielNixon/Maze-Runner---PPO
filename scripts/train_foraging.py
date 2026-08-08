from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from mazerunner_ppo.exact_env import ExactGameConfig, ExactMazeRunnerEnv
from mazerunner_ppo.foraging import ForagingRewardConfig, ForagingRewardWrapper
from mazerunner_ppo.record import SELECTION_METRICS, RecordEvalCallback

FORAGING_MILESTONE_REWARDS = (
    (60.0, 10.0),
    (120.0, 20.0),
    (180.0, 30.0),
    (240.0, 40.0),
    (300.0, 60.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the record policy for path-aware orb foraging"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("runs/record-chase/final_record_model.zip"),
        help="Record-chase checkpoint to continue from",
    )
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--foraging-envs",
        type=int,
        default=1,
        help="Number of training envs using hunger+orbs curriculum; the rest stay full Level 3",
    )
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--eval-seed", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="p90")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--ent-coef", type=float, default=0.004)
    parser.add_argument("--path-scale", type=float, default=0.35)
    parser.add_argument("--useful-orb-bonus", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("runs/foraging-chase"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def game_config(frame_skip: int) -> ExactGameConfig:
    return ExactGameConfig(
        frame_skip=frame_skip,
        max_episode_seconds=360.0,
        survival_reward=0.03,
        orb_reward=1.75,
        spike_penalty=-0.5,
        death_penalty=-5.0,
        blocked_move_penalty=-0.005,
        enemy_distance_scale=0.08,
        # Disable the old straight-line orb potential. The wrapper below uses
        # shortest reachable path distance through the visible wall layout.
        hungry_orb_distance_scale=0.0,
        hunger_orb_shape_start=0.75,
        hunger_orb_urgency_power=2.0,
        survival_milestone_rewards=FORAGING_MILESTONE_REWARDS,
    )


def make_train_env(
    rank: int,
    seed: int,
    frame_skip: int,
    foraging_envs: int,
    reward_config: ForagingRewardConfig,
):
    def factory():
        curriculum_level = 1 if rank < foraging_envs else 3
        env = ExactMazeRunnerEnv(
            config=game_config(frame_skip),
            curriculum_level=curriculum_level,
        )
        env = ForagingRewardWrapper(env, reward_config)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return factory


def make_eval_env(seed: int, frame_skip: int):
    def factory():
        env = ExactMazeRunnerEnv(
            config=game_config(frame_skip),
            curriculum_level=3,
        )
        env.reset(seed=seed)
        return Monitor(env)

    return factory


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least one")
    if not 0 <= args.foraging_envs < args.num_envs:
        raise ValueError("--foraging-envs must be >= 0 and smaller than --num-envs")
    if args.eval_episodes < 1 or args.eval_every < 1:
        raise ValueError("evaluation settings must be positive")
    if not args.resume.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    reward_config = ForagingRewardConfig(
        path_scale=args.path_scale,
        useful_orb_bonus=args.useful_orb_bonus,
    )
    reward_config.validate()

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    best_dir = args.output / "best"
    checkpoint_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)

    train_env = VecMonitor(
        SubprocVecEnv(
            [
                make_train_env(
                    rank,
                    args.seed,
                    args.frame_skip,
                    args.foraging_envs,
                    reward_config,
                )
                for rank in range(args.num_envs)
            ],
            start_method="spawn",
        )
    )
    eval_env = SubprocVecEnv(
        [make_eval_env(args.eval_seed, args.frame_skip)],
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
        name_prefix="foraging_recurrent_ppo",
    )
    eval_callback = RecordEvalCallback(
        eval_env,
        eval_freq=max(args.eval_every // args.num_envs, 1),
        n_eval_episodes=args.eval_episodes,
        seed=args.eval_seed,
        save_path=best_dir,
        log_path=args.output / "foraging_eval.jsonl",
        selection_metric=args.selection_metric,
        verbose=1,
    )

    print(
        "Foraging fine-tune starting from "
        f"{args.resume} | {args.foraging_envs}/{args.num_envs} clean-foraging envs | "
        f"path_scale={args.path_scale:g} useful_orb_bonus={args.useful_orb_bonus:g}"
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=False,
        )
        model.save(args.output / "final_foraging_model")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
