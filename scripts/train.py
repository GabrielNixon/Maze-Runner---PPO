from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from mazerunner_ppo.env import MazeRunnerEnv
from mazerunner_ppo.model import MazeFeaturesExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recurrent PPO on MazeRunnerPPO-v0")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--curriculum-level", type=int, choices=range(4), default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/default"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--single-process", action="store_true")
    return parser.parse_args()


def make_env(rank: int, seed: int, curriculum_level: int):
    def _factory():
        env = MazeRunnerEnv(curriculum_level=curriculum_level)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _factory


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    (args.output / "best").mkdir(exist_ok=True)
    (args.output / "eval").mkdir(exist_ok=True)

    factories = [
        make_env(rank, args.seed, args.curriculum_level)
        for rank in range(args.num_envs)
    ]
    if args.single_process or args.num_envs == 1:
        train_env = DummyVecEnv(factories)
    else:
        train_env = SubprocVecEnv(factories, start_method="spawn")
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([make_env(10_000, args.seed, args.curriculum_level)])
    eval_env = VecMonitor(eval_env)

    policy_kwargs = {
        "features_extractor_class": MazeFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
        "net_arch": {"pi": [128], "vf": [128]},
    }

    if args.resume is not None:
        model = RecurrentPPO.load(args.resume, env=train_env, device=args.device)
    else:
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=4,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(args.output / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25_000 // args.num_envs, 1),
        save_path=str(args.output / "checkpoints"),
        name_prefix="recurrent_ppo",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output / "best"),
        log_path=str(args.output / "eval"),
        eval_freq=max(20_000 // args.num_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=args.resume is None,
        )
        model.save(args.output / "final_model")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
