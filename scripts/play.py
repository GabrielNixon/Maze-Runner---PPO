from __future__ import annotations

import argparse

import pygame

from mazerunner_ppo.env import MazeRunnerEnv


def choose_action(keys: pygame.key.ScancodeWrapper) -> int:
    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    if up and right:
        return 2
    if down and right:
        return 4
    if down and left:
        return 6
    if up and left:
        return 8
    if up:
        return 1
    if right:
        return 3
    if down:
        return 5
    if left:
        return 7
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the symbolic environment")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--curriculum-level", type=int, choices=range(4), default=3)
    args = parser.parse_args()

    env = MazeRunnerEnv(render_mode="human", curriculum_level=args.curriculum_level)
    env.reset(seed=args.seed)
    terminated = truncated = False

    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                env.close()
                return
        action = choose_action(pygame.key.get_pressed())
        _, _, terminated, truncated, info = env.step(action)

    print(info)
    env.close()


if __name__ == "__main__":
    main()
