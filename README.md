# Maze Runner — Recurrent PPO

The goal of this repository is concrete: **train an agent on the real MazeRunner mechanics and use it to earn a legitimate Survival score that qualifies for the game's normal top-10 leaderboard modal.**

The project now has two backends:

- **Symbolic backend:** a fast Python approximation for debugging and optional pretraining.
- **Exact backend:** a headless shared library compiled from a pinned checkout of [`nadarenator/mazerunner`](https://github.com/nadarenator/mazerunner). It uses the upstream WFC maze, continuous player movement, hunger, orbs, spike timing, enemy spawning, BFS pursuit, collisions, and death rules.

The upstream source is cloned during setup and is not copied into this repository. The exact integration is pinned to commit `95f83d1a35717fe4dacbb50fc12a9b57f1575754` so training and evaluation remain reproducible.

## Install

Python 3.10 or newer is required.

```bash
git clone https://github.com/GabrielNixon/Maze-Runner---PPO.git
cd Maze-Runner---PPO
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Build the exact game backend

A C compiler and Git are required. On macOS, the Apple command-line tools are sufficient:

```bash
xcode-select --install  # only when a compiler is not already installed
python scripts/bootstrap_exact.py
```

This command:

1. Clones the pinned upstream repository into `.exact_game/mazerunner`.
2. Compiles the real game modules against a deterministic headless Raylib compatibility layer.
3. Writes `libmazerunner_exact.dylib` on macOS or `libmazerunner_exact.so` on Linux under `mazerunner_ppo/lib/`.

Validate both backends:

```bash
pytest
ruff check .
```

## Train toward the leaderboard

Do not begin by training the full game from scratch. Use the exact-game curriculum and carry the recurrent policy forward:

```bash
# Stage 1: exact movement, WFC maze, hunger and orbs
python scripts/train_exact.py \
  --curriculum-level 1 \
  --timesteps 500000 \
  --num-envs 4 \
  --output runs/exact-level1

# Stage 2: add exact spike timing and damage
python scripts/train_exact.py \
  --curriculum-level 2 \
  --timesteps 500000 \
  --num-envs 4 \
  --resume runs/exact-level1/best/best_model.zip \
  --output runs/exact-level2

# Stage 3: full Survival mode with enemies
python scripts/train_exact.py \
  --curriculum-level 3 \
  --timesteps 3000000 \
  --num-envs 4 \
  --resume runs/exact-level2/best/best_model.zip \
  --output runs/exact-full
```

Each exact environment runs in its own process because the original C game uses process-global state. `train_exact.py` therefore always uses `SubprocVecEnv`, including when `--num-envs 1` is selected.

Monitor learning:

```bash
tensorboard --logdir runs
```

## Evaluate on held-out exact-game seeds

```bash
python scripts/evaluate_exact.py \
  runs/exact-full/best/best_model.zip \
  --episodes 100 \
  --seed 10000
```

The important metrics are actual survival time, the distribution across unseen seeds, enemy-versus-hunger deaths, and whether the lower tail clears the current tenth-place leaderboard score. Maximum survival alone is not enough.

Render one symbolic view of an exact-game evaluation:

```bash
python scripts/evaluate_exact.py \
  runs/exact-full/best/best_model.zip \
  --episodes 1 \
  --render
```

## Run the policy in the real browser game

The leaderboard runner does **not** write scores or call Firestore. It patches a local checkout of the upstream web build to do only two things:

1. Send the agent the same local game-state observation used during exact training.
2. Apply the returned action through the normal WASD movement path.

All maze generation, timing, collisions, deaths, score calculation, top-10 qualification, name entry, and submission remain inside the original game.

### 1. Prepare the browser build

The upstream web build requires Emscripten and the upstream Raylib WASM library, as described in the original game's README. After those are available:

```bash
python scripts/prepare_leaderboard_build.py --force
cd .exact_game/leaderboard
source ~/emsdk/emsdk_env.sh
make web
```

### 2. Start the trained policy server

In a separate terminal from the repository root:

```bash
source .venv/bin/activate
python scripts/serve_policy.py runs/exact-full/best/best_model.zip
```

The server listens only on `ws://127.0.0.1:8765`. It keeps the recurrent state for the current run and resets it when the real game reaches game over.

### 3. Serve and open the actual web build

```bash
cd .exact_game/leaderboard/web
python -m http.server 8080
```

Open `http://localhost:8080`, confirm the browser console says that the policy server connected, and start **Survival**. The policy supplies legal movement actions while the game itself determines the score. When the run qualifies, the original top-10 modal appears and you enter your name normally.

## Observation and action contract

The symbolic, exact-C, and browser-agent backends share the same policy interface.

### Actions

| Action | Movement |
|---:|---|
| 0 | stay |
| 1 | north |
| 2 | north-east |
| 3 | east |
| 4 | south-east |
| 5 | south |
| 6 | south-west |
| 7 | west |
| 8 | north-west |

### Observation

```python
{
    "grid": float32[7, 19, 19],
    "stats": float32[14],
}
```

Grid channels represent walls, orbs, spike locations, active spikes, warning spikes, active enemies, and frozen/materialising enemies. Scalar features contain hunger, cyclic spike phase, enemy load, nearest-enemy distance, and the previous action.

## Project layout

```text
exact_game/
├── bridge/                  # deterministic headless Raylib shim and exact C API
└── web/                     # browser observation/input hook copied into a local upstream clone
mazerunner_ppo/
├── env.py                   # symbolic Gymnasium backend
├── exact_env.py             # ctypes wrapper around the real C simulation
├── model.py                 # CNN feature extractor for RecurrentPPO
└── ...
scripts/
├── bootstrap_exact.py       # clone pinned upstream source and build shared library
├── train_exact.py
├── evaluate_exact.py
├── prepare_leaderboard_build.py
├── serve_policy.py
├── train.py                 # symbolic training
└── evaluate.py              # symbolic evaluation
```

## Integrity of leaderboard runs

This repository deliberately does not include score injection, direct leaderboard writes, altered death conditions, disabled enemies, or modified scoring. The agent must play through the normal update loop and qualify according to the leaderboard's own comparison logic. Curriculum simplifications exist only in the headless training backend; the browser leaderboard build always runs full Survival mode.

## Acknowledgement

Game concept and upstream implementation: [`nadarenator/mazerunner`](https://github.com/nadarenator/mazerunner). This repository contains the reinforcement-learning environment, bridge, training code, and optional browser control hook; it does not redistribute the upstream game source.
