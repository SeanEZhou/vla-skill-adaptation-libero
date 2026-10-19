"""M2: visual inspection of one demo: image strip (agentview | wrist) + action traces.

    uv run python scripts/plot_dataset_episode.py --episode 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vla_sim_poc.action_adapter import ACTION_LAYOUT  # noqa: E402
from vla_sim_poc.config import PROJECT_ROOT  # noqa: E402
from vla_sim_poc.video import save_png  # noqa: E402

# Reference palette: categorical slots 1-3 (blue, orange, aqua); neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT / "data/lerobot/libero90_t56"))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results/dataset"))
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/libero90_t56_butter_tray", root=args.root)
    rows = np.nonzero(np.asarray(ds.hf_dataset["episode_index"]) == args.episode)[0]
    acts = np.stack([ds.hf_dataset[int(r)]["action"] for r in rows])
    out = Path(args.out)

    tiles = []
    for t in np.linspace(0, len(rows) - 1, args.n_frames).astype(int):
        item = ds[int(rows[t])]
        pair = [(item[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                for k in ("observation.images.image", "observation.images.image2")]
        tiles.append(np.concatenate(pair, axis=0))
    save_png(np.concatenate(tiles, axis=1), out / f"episode_{args.episode:03d}_strip.png")

    t = np.arange(len(acts)) / ds.meta.fps
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.2), sharex=True, facecolor=SURFACE)
    groups = [("position delta (x 5 cm)", [0, 1, 2]), ("rotation delta (x 0.5 rad)", [3, 4, 5]), ("gripper (-1 open, +1 close)", [6])]
    for ax, (title, dims) in zip(axes, groups):
        ax.set_facecolor(SURFACE)
        for j, d in enumerate(dims):
            ax.plot(t, acts[:, d], color=SERIES[j], lw=2, label=ACTION_LAYOUT[d])
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.tick_params(colors=INK_2, length=0, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        if len(dims) > 1:
            ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right", labelcolor=INK)
    axes[-1].set_xlabel("time (s), 20 Hz", color=INK_2, fontsize=9)
    fig.suptitle(f"Demo actions, episode {args.episode}: \"{ds.meta.tasks.index[0]}\"", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / f"episode_{args.episode:03d}_actions.png", dpi=150, facecolor=SURFACE)
    print(f"saved {out}/episode_{args.episode:03d}_strip.png and _actions.png")


if __name__ == "__main__":
    main()
