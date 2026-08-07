from __future__ import annotations

import numpy as np


def make_frame(grid: np.ndarray, hunger: float) -> np.ndarray:
    cell, n = 20, grid.shape[1]
    frame = np.full((n * cell + 30, n * cell, 3), (15, 12, 16), np.uint8)
    colors = (
        (90, 80, 70), (60, 205, 90), (145, 135, 110),
        (225, 65, 50), (230, 165, 45), (190, 35, 35), (130, 95, 220),
    )
    for row in range(n):
        for col in range(n):
            y, x = row * cell, col * cell
            frame[y:y + cell - 1, x:x + cell - 1] = (35, 31, 28)
            for channel, color in enumerate(colors):
                if grid[channel, row, col]:
                    frame[y + 4:y + cell - 4, x + 4:x + cell - 4] = color
    mid = n // 2 * cell
    frame[mid + 5:mid + 15, mid + 5:mid + 15] = (235, 210, 85)
    fill = int((n * cell - 20) * hunger)
    frame[-20:-10, 10:10 + fill] = (210, 150, 45)
    return frame
