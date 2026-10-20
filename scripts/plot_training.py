"""Training curves (train / held-out-episode loss) above checkpoint closed-loop selection success.

    uv run python scripts/plot_training.py --run outputs/train/expert_only_s0 \
        --selection results/selection/expert_only_s0 --out results/plots/training_expert_only.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vla_sim_poc.metrics import proportion  # noqa: E402

TRAIN_C, VAL_C = "#2a78d6", "#eb6834"  # reference palette slots 1-2
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.tick_params(colors=INK_2, length=0, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Fine-tuning the action expert on 45 demos")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (Path(args.run) / "train_log.jsonl").read_text().splitlines()]
    tr = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    va = [(r["step"], r["val_loss"], r["val_sem"]) for r in rows if "val_loss" in r]

    sel = []
    for d in sorted(Path(args.selection).glob("step*")):
        eps = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]
        sel.append((int(d.name[4:]), proportion([e["success"] for e in eps])))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [3, 2]})
    style(a1)
    a1.plot(*zip(*tr), color=TRAIN_C, lw=2, label="train loss (45 demos)")
    vs, vl, ve = zip(*va)
    a1.plot(vs, vl, "o-", color=VAL_C, lw=2, ms=5, mec=SURFACE, mew=1.5, label="held-out-demo loss (5 demos)")
    a1.set_yscale("log")
    a1.set_ylabel("flow-matching loss (log)", color=INK_2, fontsize=9)
    a1.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right", bbox_to_anchor=(1.0, 0.62))
    a1.annotate(f"untrained: {vl[0]:.2f}", (vs[0], vl[0]), xytext=(8, -2), textcoords="offset points", fontsize=8, color=INK_2)

    style(a2)
    for step, p in sel:
        a2.plot([step, step], [p.ci_low, p.ci_high], color=TRAIN_C, lw=2, solid_capstyle="round")
        a2.plot(step, p.rate, "o", ms=8, color=TRAIN_C, mec=SURFACE, mew=2)
        a2.annotate(f"{p.successes}/{p.n}", (step, p.rate), xytext=(8, -3), textcoords="offset points", fontsize=8, color=INK)
    a2.set_ylim(-0.02, 1.05)
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.set_ylabel("closed-loop success\n(20 selection layouts)", color=INK_2, fontsize=9)
    a2.set_xlabel("training step (batch 32)", color=INK_2, fontsize=9)
    fig.suptitle(args.title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"plot -> {args.out}")


if __name__ == "__main__":
    main()
