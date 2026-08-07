# Five-minute record chase

This phase starts from a trained full-game checkpoint and optimizes for a single long Survival run rather than average PPO reward.

The current held-out full-game baseline that motivated this phase was:

- mean survival: 32.95 s
- median survival: 24.38 s
- maximum survival: 141.69 s
- mean orbs: 9.2
- most deaths were from hunger rather than enemies

The target is 300 seconds.

## What changes during fine-tuning

`train_record.py` keeps the exact game mechanics unchanged and changes only the training objective:

- survival reward is slightly stronger;
- orb reward is reduced so it does not dominate the leaderboard objective;
- food-distance shaping becomes progressively stronger as hunger falls;
- one-time training bonuses are paid at 60, 120, 180, 240, and 300 seconds;
- the resumed PPO policy uses a lower learning rate and longer discount horizon;
- checkpoints are evaluated on fixed held-out seeds and selected by survival statistics rather than episode reward.

The default checkpoint selector is **p90 survival time**. This encourages a strong upper tail without selecting a model from one lucky maximum. Every evaluation also records median, p95, max, orb count, death causes, and the fraction of runs clearing each survival milestone.

## Start the record phase

The default command resumes the full-game best checkpoint:

```bash
python scripts/train_record.py \
  --resume runs/exact-full/best/best_model.zip \
  --timesteps 5000000 \
  --num-envs 4 \
  --output runs/record-chase
```

Defaults:

```text
learning rate   5e-5
gamma           0.999
gae lambda      0.97
entropy coef    0.005
evaluation      50 episodes every 100k training steps
evaluation seed 20000
selection       p90 survival
```

The trainer evaluates the starting checkpoint before the first gradient update. Results are appended to:

```text
runs/record-chase/record_eval.jsonl
```

The best survival-tail checkpoint is saved as:

```text
runs/record-chase/best/best_record_model.zip
```

Its metrics are saved beside it in `best_record_metrics.json`.

## Watch the run

```bash
tensorboard --logdir runs/record-chase/tensorboard
```

The important TensorBoard series are under `record/`:

```text
record/median
record/p90
record/p95
record/max
record/over_60s
record/over_120s
record/over_180s
record/over_240s
record/over_300s
```

For this objective, p90/p95 and milestone hit rates matter more than mean reward.

## Final held-out evaluation

Do not use the training-evaluation seeds for the final decision. Evaluate the best record model on a fresh block, for example:

```bash
python scripts/evaluate_exact.py \
  runs/record-chase/best/best_record_model.zip \
  --episodes 200 \
  --seed 50000 \
  --curriculum-level 3
```

`evaluate_exact.py` reports p90, p95, milestone rates, and death counts in addition to the original mean/median/max statistics.

Once the policy starts producing 300-second exact-game runs, use the existing browser policy server and normal leaderboard-enabled game build. The policy supplies movement actions only; the original game owns scoring and leaderboard qualification.
