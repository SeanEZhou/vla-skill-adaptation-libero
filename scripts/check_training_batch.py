"""M2 exit gate: a training batch loads, forwards through the checkpoint, gives a finite loss, and its
action targets are the right 50-step chunks.

Also measures the checkpoint's flow-matching loss on the new task vs a task it was trained on
(libero_object task 0 from ``lerobot/libero``), with the checkpoint's own normalization and
identical noise/time draws, so the two numbers are directly comparable.

    uv run python scripts/check_training_batch.py
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from vla_sim_poc.config import PROJECT_ROOT, load_config
from vla_sim_poc.dataset import ACTION_KEY, episodes_for_task
from vla_sim_poc.policy import load_policy

LIBERO_REPO, LIBERO_REV = "lerobot/libero", "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
LIBERO_OBJECT_T0_TASK_INDEX = 24  # "pick up the alphabet soup and place it in the basket"


def make_ds(repo_id, root, revision, episodes, cfg):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    return LeRobotDataset(repo_id, root=root, revision=revision, episodes=episodes,
                          delta_timestamps=resolve_delta_timestamps(cfg, meta))


@torch.no_grad()
def mean_loss(bundle, ds, n_batches, batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[: n_batches * batch_size].tolist()
    chunk, dim = bundle.config.chunk_size, bundle.config.max_action_dim
    losses = []
    for b in range(n_batches):
        items = [ds[i] for i in idx[b * batch_size:(b + 1) * batch_size]]
        batch = torch.utils.data.default_collate(items)
        batch = bundle.preprocessor(batch)
        # identical noise / flow time for every dataset -> losses are comparable
        gb = torch.Generator(device=batch[ACTION_KEY].device).manual_seed(seed + b)
        noise = torch.randn((batch_size, chunk, dim), generator=gb, device=batch[ACTION_KEY].device)
        time = torch.rand((batch_size,), generator=gb, device=batch[ACTION_KEY].device) * 0.999 + 0.001
        loss, _ = bundle.policy.forward(batch, noise=noise, time=time)
        losses.append(loss.item())
    return float(np.mean(losses)), float(np.std(losses) / np.sqrt(len(losses)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=str(PROJECT_ROOT / "data/lerobot/libero90_t56"))
    ap.add_argument("--n-batches", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results/dataset/batch_check.json"))
    args = ap.parse_args()
    cfg = load_config("eval.yaml", "baseline.yaml", "selected_task.yaml")
    bundle = load_policy(cfg["checkpoint"], cfg["policy"])
    pcfg = bundle.config
    ours_root = Path(args.ours)
    split = json.loads((ours_root / "vla_sim_poc/split.json").read_text())
    ours = make_ds("local/libero90_t56_butter_tray", ours_root, None, split["train_episodes"], pcfg)

    # --- target check: item t's action chunk == stored actions t..t+49, padded past episode end
    raw = ours.hf_dataset
    ep_col, act_col = np.asarray(raw["episode_index"]), np.stack(raw["action"])
    checks = []
    for i in [0, 37, len(ours) - 1]:
        item = ours[i]
        e, fi = int(item["episode_index"]), int(item["frame_index"])
        ep_actions = act_col[ep_col == e]
        valid = ~item["action_is_pad"].numpy()
        n_valid = min(pcfg.chunk_size, len(ep_actions) - fi)
        np.testing.assert_allclose(item[ACTION_KEY].numpy()[valid], ep_actions[fi:fi + n_valid], atol=1e-6)
        assert valid.sum() == n_valid
        checks.append({"index": i, "episode": e, "frame": fi, "valid_steps": int(n_valid)})

    batch = torch.utils.data.default_collate([ours[i] for i in range(args.batch_size)])
    shapes = {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    pre = bundle.preprocessor(copy.deepcopy(batch))
    loss, loss_dict = bundle.policy.forward(pre)
    assert torch.isfinite(loss), loss

    # --- offline loss: new task vs a training task, checkpoint stats, same noise draws
    known_eps = episodes_for_task(LIBERO_REPO, LIBERO_REV, LIBERO_OBJECT_T0_TASK_INDEX)[:10]
    known = make_ds(LIBERO_REPO, None, LIBERO_REV, known_eps, pcfg)
    assert {"observation.images.image", "observation.images.image2"} <= set(known.meta.features)
    l_new = mean_loss(bundle, ours, args.n_batches, args.batch_size, args.seed)
    l_known = mean_loss(bundle, known, args.n_batches, args.batch_size, args.seed)

    report = {
        "batch_shapes": shapes,
        "action_chunk_checks": checks,
        "single_batch_loss": loss.item(),
        "loss_dict": {k: float(v) for k, v in loss_dict.items()},
        "offline_flow_loss_ckpt_stats": {
            "target_task_libero90_t56_train_split": {"mean": l_new[0], "sem": l_new[1]},
            "known_task_libero_object_t0": {"mean": l_known[0], "sem": l_known[1]},
            "n_batches": args.n_batches, "batch_size": args.batch_size,
        },
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
