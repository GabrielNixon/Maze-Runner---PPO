from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
import websockets

GRID_FLOATS = 7 * 19 * 19
TOTAL_FLOATS = GRID_FLOATS + 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a trained recurrent PPO policy to the browser agent build"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


async def serve(args: argparse.Namespace) -> None:
    model = RecurrentPPO.load(args.model, device=args.device)

    async def handler(websocket) -> None:
        state = None
        episode_start = np.ones((1,), dtype=bool)
        print("Browser connected")
        try:
            async for message in websocket:
                if isinstance(message, str):
                    if message == "reset":
                        state = None
                        episode_start[:] = True
                    continue
                values = np.frombuffer(message, dtype=np.float32)
                if values.size != TOTAL_FLOATS:
                    print(f"Ignoring observation with {values.size} floats")
                    continue
                observation = {
                    "grid": values[:GRID_FLOATS].reshape(7, 19, 19).copy(),
                    "stats": values[GRID_FLOATS:].reshape(14).copy(),
                }
                action, state = model.predict(
                    observation,
                    state=state,
                    episode_start=episode_start,
                    deterministic=not args.stochastic,
                )
                episode_start[:] = False
                await websocket.send(bytes([int(action)]))
        finally:
            print("Browser disconnected")

    async with websockets.serve(
        handler,
        args.host,
        args.port,
        max_size=TOTAL_FLOATS * 4 + 1024,
    ):
        print(f"Policy server listening on ws://{args.host}:{args.port}")
        await asyncio.Future()


def main() -> None:
    args = parse_args()
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
