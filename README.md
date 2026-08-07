# Maze Runner — Recurrent PPO

A fast reinforcement-learning environment and recurrent PPO agent inspired by the mechanics of [nadarenator/mazerunner](https://github.com/nadarenator/mazerunner).

The original game is a real-time C/Raylib survival game with a rolling procedurally generated maze, hunger, collectible food orbs, periodic spike traps, and BFS-controlled enemies. This repository provides an **original symbolic, headless reimplementation of those mechanics for RL research**. It does not copy or redistribute the upstream source code.

## What is included

- A Gymnasium environment with nine discrete movement actions: stay plus eight directions.
- A rolling maze buffer. Terrain that leaves the window is discarded and newly generated terrain enters from the opposite edge.
- Hunger decay, food orbs, warning/active spike phases, frozen enemy spawns, and BFS pursuit.
- A compact seven-channel player-centred observation rather than screenshots.
- A custom CNN feature extractor followed by the recurrent policy in `sb3-contrib`'s `RecurrentPPO`.
- Parallel training, checkpointing, TensorBoard logging, evaluation, rendering, deterministic seeds, tests, and CI.
- Four curriculum levels:
  - `0`: navigation only
  - `1`: hunger and orbs
  - `2`: spikes
  - `3`: full game with enemies

## Environment

### Action space

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

Diagonal movement cannot cut through the corner of two walls.

### Observation space

The observation is a dictionary:

```python
{
    "grid": float32[7, 19, 19],
    "stats": float32[14],
}
```

The grid channels are walls, orbs, spike locations, active spikes, warning spikes, active enemies, and frozen/materialising enemies. Scalar state contains hunger, cyclic spike timing, enemy information, and the previous action.

### Reward

The default shaped training reward is:

- `+0.01` for each survived decision step
- `+1.0` for collecting an orb
- `-0.25` for spike damage
- `-10.0` on death
- a tiny penalty for walking into a wall

Evaluation should focus on the unshaped metrics reported in `info`: survival time and orb count.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/GabrielNixon/Maze-Runner---PPO.git
cd Maze-Runner---PPO
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Apple Silicon, PyTorch normally uses the MPS backend automatically when available. Training also works on CPU.

## Validate the environment

```bash
pytest
ruff check .
```

You can manually play the symbolic environment:

```bash
python scripts/play.py --seed 7 --curriculum-level 3
```

Use WASD or the arrow keys. Diagonal movement is produced by holding two directions together.

## Train

Start with the complete environment:

```bash
python scripts/train.py \
  --timesteps 1000000 \
  --num-envs 8 \
  --curriculum-level 3 \
  --output runs/full
```

For a MacBook or for debugging, use fewer environments and one process:

```bash
python scripts/train.py \
  --timesteps 100000 \
  --num-envs 2 \
  --single-process \
  --output runs/smoke-test
```

A more reliable curriculum is to train successive stages and resume from the previous checkpoint:

```bash
python scripts/train.py --timesteps 200000 --curriculum-level 1 --output runs/level1
python scripts/train.py --timesteps 300000 --curriculum-level 2 \
  --resume runs/level1/final_model.zip --output runs/level2
python scripts/train.py --timesteps 1000000 --curriculum-level 3 \
  --resume runs/level2/final_model.zip --output runs/level3
```

TensorBoard logs are written under the selected output folder:

```bash
tensorboard --logdir runs
```

## Evaluate

```bash
python scripts/evaluate.py runs/full/best/best_model.zip \
  --episodes 50 \
  --seed 10000
```

Render a single evaluation run:

```bash
python scripts/evaluate.py runs/full/best/best_model.zip \
  --episodes 1 \
  --render
```

The evaluator keeps and resets the recurrent LSTM state correctly between episodes and prints JSON containing the mean, median, and maximum survival time plus orb statistics.

## Minimal Python usage

```python
from mazerunner_ppo import MazeRunnerEnv

env = MazeRunnerEnv(curriculum_level=3)
obs, info = env.reset(seed=42)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
print(info)
```

## Relationship to the browser game

This first version trains against a purpose-built symbolic environment, not pixels from the browser and not the upstream C executable. That is deliberate: it makes training fast enough to run many environments in parallel and exposes the state required for recurrent PPO.

The next exact-game integration step would be to add a small C API to the upstream game—`reset`, `step`, `observe`, `reward`, and `done`—and compile it as a shared library. The policy and training scripts here can then remain largely unchanged while the environment backend is replaced. Because the upstream repository currently has no explicit license file, this repository does not vendor or modify its source.

## Project structure

```text
mazerunner_ppo/
├── env.py          # rolling symbolic environment and renderer
├── model.py        # CNN feature extractor for RecurrentPPO
└── __init__.py     # Gymnasium registration
scripts/
├── train.py
├── evaluate.py
└── play.py
tests/
├── test_env.py
└── test_model.py
```

## Acknowledgement

Game concept and upstream implementation: [nadarenator/mazerunner](https://github.com/nadarenator/mazerunner). This project is an independent RL environment inspired by its publicly described mechanics.
