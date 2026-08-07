# Exact game integration

This directory contains only the reinforcement-learning bridge and browser hook. It does not contain a copy of the upstream MazeRunner source.

`python scripts/bootstrap_exact.py` clones the pinned upstream commit into the ignored `.exact_game/mazerunner` directory and compiles these bridge files together with the upstream game modules.

`python scripts/prepare_leaderboard_build.py` creates a separate ignored checkout, copies the browser hook into it, and applies narrowly scoped source edits. The hook sends observations to a local policy server and returns movement actions. It does not modify scoring, death conditions, top-10 qualification, or leaderboard submission.
