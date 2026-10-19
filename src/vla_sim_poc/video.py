"""Rollout video helpers. Frames are raw LIBERO renders, rotated upright for human viewing."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def viewer_frame(raw_obs: dict) -> np.ndarray:
    """Agentview | wrist side by side, rotated 180 deg (LIBERO renders upside down)."""
    pix = raw_obs["pixels"]
    return np.concatenate([pix["image"][::-1, ::-1], pix["image2"][::-1, ::-1]], axis=1)


def write_mp4(frames: list[np.ndarray], path: str | Path, fps: int = 20) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=1)


def save_png(frame: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)
