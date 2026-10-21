"""Training helpers built on LeRobot components (model, processors, optimizer/scheduler presets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

ACTION_KEY = "action"


class SingleFrameImages(torch.utils.data.Dataset):
    """Give every dataset the same image layout. With one observation step, LeRobot returns
    image-backed frames as (1, C, H, W) but video-backed frames as (C, H, W); mixing the two in a
    batch fails to collate. Drop the singleton time dim (SmolVLA takes the last frame of a 5-D tensor
    anyway, so values are unchanged)."""

    def __init__(self, ds):
        self.ds = ds
        self.meta = ds.meta
        self.hf_dataset = ds.hf_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        for k in self.meta.camera_keys:
            if item[k].ndim == 4 and item[k].shape[0] == 1:
                item[k] = item[k][0]
        return item


def reset_video_decoders() -> None:
    """Drop LeRobot's per-process cache of torchcodec decoders so the next read opens a fresh one."""
    from lerobot.datasets import video_utils

    video_utils._default_decoder_cache.clear()


class RetryOnDecodeError(torch.utils.data.Dataset):
    """Training-time guard for video-decode failures (torchcodec "Could not push packet to decoder").
    The failure is state-dependent, not bad data: every file decodes cleanly with ffmpeg, and the
    failing frame decodes fine in a fresh process. A *cached* decoder in a long-lived worker gets
    into a bad state after many random seeks and keeps failing. So: reset the decoder cache and retry
    the same sample; if it still fails, substitute a random other sample and log it."""

    def __init__(self, ds, retries: int = 3, seed: int = 0):
        self.ds, self.retries = ds, retries
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i):
        for attempt in range(self.retries + 1):
            try:
                return self.ds[i]
            except RuntimeError as e:
                reset_video_decoders()
                j = i if attempt < self.retries else int(self.rng.integers(len(self.ds)))
                print(f"[data] decode error on sample {i} (attempt {attempt + 1}): {str(e)[:120]}; "
                      f"{'retrying' if j == i else f'substituting sample {j}'}", flush=True)
                i = j
        return self.ds[i]


def make_dataset(repo_id: str, root: str | Path | None, episodes: list[int] | None, policy_cfg, revision=None):
    """LeRobotDataset yielding (frame t, actions t..t+chunk-1) samples, as SmolVLA trains on."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    if policy_cfg.n_obs_steps != 1:
        raise NotImplementedError("SingleFrameImages assumes n_obs_steps == 1")
    return SingleFrameImages(LeRobotDataset(repo_id, root=root, revision=revision, episodes=episodes,
                                            delta_timestamps=resolve_delta_timestamps(policy_cfg, meta)))


@torch.no_grad()
def fixed_noise_loss(policy, preprocessor, ds, n_batches: int, batch_size: int, seed: int = 0) -> tuple[float, float]:
    """Flow-matching loss with fixed samples and fixed noise/time draws, so values are comparable
    across checkpoints and datasets. Returns (mean, standard error)."""
    was_training = policy.training
    policy.eval()
    g = torch.Generator().manual_seed(seed)
    n = min(len(ds), n_batches * batch_size)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    chunk, dim = policy.config.chunk_size, policy.config.max_action_dim
    losses = []
    for b in range(0, n, batch_size):
        items = [ds[i] for i in idx[b:b + batch_size]]
        batch = preprocessor(torch.utils.data.default_collate(items))
        dev, bs = batch[ACTION_KEY].device, len(items)
        gb = torch.Generator(device=dev).manual_seed(seed + b)
        noise = torch.randn((bs, chunk, dim), generator=gb, device=dev)
        time = torch.rand((bs,), generator=gb, device=dev) * 0.999 + 0.001
        loss, _ = policy.forward(batch, noise=noise, time=time)
        losses.append(loss.item())
    policy.train(was_training)
    return float(np.mean(losses)), float(np.std(losses) / np.sqrt(max(1, len(losses))))


def count_params(policy) -> dict[str, Any]:
    trainable = [p for p in policy.parameters() if p.requires_grad]
    return {
        "total_M": round(sum(p.numel() for p in policy.parameters()) / 1e6, 1),
        "trainable_M": round(sum(p.numel() for p in trainable) / 1e6, 1),
        "trainable_dtypes": sorted({str(p.dtype) for p in trainable}),
    }
