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

TRAIN_C, VAL_C, RET_C = "#2a78d6", "#eb6834", "#1baf7a"  # reference palette slots 1-3
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
    ap.add_argument("--selection", required=True, help="dir of step*/ runs on the target selection set")
    ap.add_argument("--retention-selection", default=None, help="dir of step*/ runs on the retention selection set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Fine-tuning the action expert on 45 demos")
    ap.add_argument("--train-label", default="train loss (45 demos)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (Path(args.run) / "train_log.jsonl").read_text().splitlines()]
    tr = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    va = [(r["step"], r["val_loss"], r["val_sem"]) for r in rows if "val_loss" in r]

    def selection(root):
        out = []
        for d in sorted(Path(root).glob("step*")):
            eps = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]
            out.append((int(d.name[4:]), proportion([e["success"] for e in eps])))
        return out

    sels = [("target task", TRAIN_C, selection(args.selection))]
    if args.retention_selection:
        sels.append(("retention (LIBERO-Object)", RET_C, selection(args.retention_selection)))
    ret = [(r["step"], r["retention_loss"]) for r in rows if "retention_loss" in r]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [3, 2]})
    style(a1)
    a1.plot(*zip(*tr), color=TRAIN_C, lw=2, label=args.train_label)
    vs, vl, ve = zip(*va)
    a1.plot(vs, vl, "o-", color=VAL_C, lw=2, ms=5, mec=SURFACE, mew=1.5, label="held-out-demo loss (5 demos)")
    if ret:
        a1.plot(*zip(*ret), "o-", color=RET_C, lw=2, ms=5, mec=SURFACE, mew=1.5, label="known-task loss (LIBERO-Object t0)")
    a1.set_yscale("log")
    a1.set_ylabel("flow-matching loss (log)", color=INK_2, fontsize=9)
    a1.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right", bbox_to_anchor=(1.0, 0.62))
    a1.annotate(f"untrained: {vl[0]:.2f}", (vs[0], vl[0]), xytext=(8, -2), textcoords="offset points", fontsize=8, color=INK_2)

    style(a2)
    span = max(r["step"] for r in rows)
    for j, (name, color, sel) in enumerate(sels):
        dx = (j - (len(sels) - 1) / 2) * 0.02 * span
        for k, (step, p) in enumerate(sel):
            a2.plot([step + dx] * 2, [p.ci_low, p.ci_high], color=color, lw=2, solid_capstyle="round")
            a2.plot(step + dx, p.rate, "o", ms=8, color=color, mec=SURFACE, mew=2, label=name if k == 0 else None)
            a2.annotate(f"{p.successes}/{p.n}", (step + dx, p.rate), xytext=(6, 3 if j == 0 else -11),
                        textcoords="offset points", fontsize=7, color=INK)
    if len(sels) > 1:
        a2.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    a2.set_ylim(-0.02, 1.05)
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.set_ylabel("closed-loop success\n(20-episode selection sets)", color=INK_2, fontsize=9)
    a2.set_xlabel("training step (batch 32)", color=INK_2, fontsize=9)
    fig.suptitle(args.title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"plot -> {args.out}")


if __name__ == "__main__":
    main()
