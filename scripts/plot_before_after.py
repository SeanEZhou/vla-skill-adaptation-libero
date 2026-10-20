"""Before/after fine-tuning: target task and retention on a known suite, Wilson 95% CIs.

    uv run python scripts/plot_before_after.py --out results/plots/before_after_expert_only.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vla_sim_poc.config import PROJECT_ROOT  # noqa: E402
from vla_sim_poc.metrics import proportion  # noqa: E402

BEFORE, AFTER = "#2a78d6", "#eb6834"  # reference palette slots 1-2
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(d):
    return proportion([json.loads(l)["success"] for l in (PROJECT_ROOT / d / "episodes.jsonl").read_text().splitlines() if l.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after-name", default="expert-only fine-tune")
    ap.add_argument("--after-dir", default="results/finetuned/expert_only_s0")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results/plots/before_after_expert_only.png"))
    args = ap.parse_args()
    rows = [
        ("Target: LIBERO-90 task 56\nbutter -> tray (50 held-out states)", "results/baseline/libero_90_t56", f"{args.after_dir}/libero_90_t56"),
        ("Retention: LIBERO-Object\n10 known tasks x 5 states", "results/validation/libero_object_n10", f"{args.after_dir}/libero_object_n10"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 2.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (label, b, a) in enumerate(rows):
        for off, d, c, name in ((-0.13, b, BEFORE, "before (smolvla_libero)"), (0.13, a, AFTER, args.after_name)):
            p = load(d)
            ax.plot([p.ci_low, p.ci_high], [i + off] * 2, color=c, lw=2, solid_capstyle="round")
            ax.plot(p.rate, i + off, "o", ms=8, color=c, mec=SURFACE, mew=2, label=name if i == 0 else None)
            ax.annotate(f"{p.successes}/{p.n}", (p.ci_high, i + off), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("closed-loop success (dot) with Wilson 95% CI", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="x", color=GRID, lw=1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"plot -> {args.out}")


if __name__ == "__main__":
    main()
