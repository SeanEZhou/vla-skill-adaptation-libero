"""Training helpers built on LeRobot components (model, processors, optimizer/scheduler presets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

ACTION_KEY = "action"


def make_dataset(repo_id: str, root: str | Path | None, episodes: list[int] | None, policy_cfg, revision=None):
    """LeRobotDataset yielding (frame t, actions t..t+chunk-1) samples, as SmolVLA trains on."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    return LeRobotDataset(repo_id, root=root, revision=revision, episodes=episodes,
                          delta_timestamps=resolve_delta_timestamps(policy_cfg, meta))


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
