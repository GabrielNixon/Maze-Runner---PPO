from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from mazerunner_ppo.exact_env import ExactGameConfig, ExactMazeRunnerEnv
from mazerunner_ppo.model import MazeFeaturesExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recurrent PPO on the exact C game")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--curriculum-level", type=int, choices=range(4), default=3)
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/exact"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def make_env(rank: int, seed: int, level: int, frame_skip: int):
    def factory():
        config = ExactGameConfig(frame_skip=frame_skip)
        env = ExactMazeRunnerEnv(config=config, curriculum_level=level)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return factory


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least one")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "best", "eval"):
        (args.output / name).mkdir(exist_ok=True)

    factories = [
        make_env(rank, args.seed, args.curriculum_level, args.frame_skip)
        for rank in range(args.num_envs)
    ]
    train_env = VecMonitor(SubprocVecEnv(factories, start_method="spawn"))
    eval_env = VecMonitor(
        SubprocVecEnv(
            [make_env(10_000, args.seed, args.curriculum_level, args.frame_skip)],
            start_method="spawn",
        )
    )

    policy_kwargs = {
        "features_extractor_class": MazeFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
        "net_arch": {"pi": [128], "vf": [128]},
    }
    if args.resume:
        model = RecurrentPPO.load(args.resume, env=train_env, device=args.device)
    else:
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=512,
            n_epochs=4,
            gamma=0.997,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.015,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(args.output / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // args.num_envs, 1),
            save_path=str(args.output / "checkpoints"),
            name_prefix="exact_recurrent_ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(args.output / "best"),
            log_path=str(args.output / "eval"),
            eval_freq=max(50_000 // args.num_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
        ),
    ]
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=args.resume is None,
        )
        model.save(args.output / "final_model")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
