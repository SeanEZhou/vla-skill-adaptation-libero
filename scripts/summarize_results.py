"""Aggregate per-episode results into tables + a success-rate plot, and compare arms pairwise.

    # one arm (e.g. screening)
    uv run python scripts/summarize_results.py --arm screening=results/screening/libero_90 \
        --plot results/plots/screening.png --md results/screening/libero_90/summary.md
    # baseline vs fine-tuned, paired on identical init states
    uv run python scripts/summarize_results.py --arm baseline=results/baseline/X --arm finetuned=results/finetuned/X
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vla_sim_poc.metrics import paired_comparison, proportion  # noqa: E402

# Categorical slots 1-2 of the validated reference palette (blue, orange); ink stays neutral.
ARM_COLORS = ["#2a78d6", "#eb6834"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_arm(d: Path) -> list[dict]:
    return [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]


def pair_key(r: dict) -> tuple:
    return (r["suite"], r["task_id"], r["init_state_id"], r["seed"] if r["init_state_id"] is None else None)


def table(arms: dict[str, list[dict]]) -> tuple[str, dict]:
    tasks = sorted({(r["suite"], r["task_id"], r["instruction"]) for rows in arms.values() for r in rows})
    names = list(arms)
    lines = ["| suite | task | instruction | " + " | ".join(f"{n} success (95% CI)" for n in names) + " | mean steps |",
             "|---|---|---|" + "---|" * len(names) + "---|"]
    stats = {}
    for suite, tid, instr in tasks:
        cells = []
        for n in names:
            rs = [r for r in arms[n] if r["suite"] == suite and r["task_id"] == tid]
            p = proportion([r["success"] for r in rs])
            stats[(n, suite, tid)] = p
            cells.append(f"{p.successes}/{p.n} = {p.rate:.0%} ({p.ci_low:.0%}-{p.ci_high:.0%})" if p.n else "-")
        steps = [r["steps"] for rs in arms.values() for r in rs if r["suite"] == suite and r["task_id"] == tid]
        lines.append(f"| {suite} | {tid} | {instr} | " + " | ".join(cells) + f" | {sum(steps) / len(steps):.0f} |")
    for n in names:
        p = proportion([r["success"] for r in arms[n]])
        lines.append(f"| **all** | | | " + " | ".join(
            f"**{p.successes}/{p.n} = {p.rate:.0%} ({p.ci_low:.0%}-{p.ci_high:.0%})**" if m == n else "" for m in names) + " | |")
    return "\n".join(lines), {"tasks": tasks, "stats": stats}


def plot(arms: dict[str, list[dict]], info: dict, path: Path, title: str) -> None:
    tasks, stats, names = info["tasks"], info["stats"], list(arms)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(tasks) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    offsets = [0.0] if len(names) == 1 else [-0.15, 0.15]
    for j, n in enumerate(names):
        for i, (suite, tid, _) in enumerate(tasks):
            p = stats.get((n, suite, tid))
            if not p or not p.n:
                continue
            y = i + offsets[j]
            ax.plot([p.ci_low, p.ci_high], [y, y], color=ARM_COLORS[j], lw=2, solid_capstyle="round")
            ax.plot(p.rate, y, "o", ms=8, color=ARM_COLORS[j], mec=SURFACE, mew=2, label=n if i == 0 else None)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f"{s.replace('libero_', '')} t{t}: {ins[:60]}" for s, t, ins in tasks], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("closed-loop success rate (dot) with Wilson 95% CI", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="x", color=GRID, lw=1)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    if len(names) > 1:
        ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=results_dir (max 2)")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--title", default="Closed-loop success by task")
    args = ap.parse_args()
    arms = {}
    for a in args.arm:
        n, d = a.split("=", 1)
        arms[n] = load_arm(Path(d))
    md, info = table(arms)
    if len(arms) == 2:
        (na, ra), (nb, rb) = arms.items()
        a_by, b_by = {pair_key(r): r for r in ra}, {pair_key(r): r for r in rb}
        keys = sorted(set(a_by) & set(b_by), key=str)
        c = paired_comparison([a_by[k]["success"] for k in keys], [b_by[k]["success"] for k in keys])
        md += (f"\n\n**Paired ({c.n_pairs} identical initial states):** {na} {c.rate_a:.0%} -> {nb} {c.rate_b:.0%}, "
               f"diff {c.diff:+.0%} (bootstrap 95% CI {c.bootstrap_ci[0]:+.0%} to {c.bootstrap_ci[1]:+.0%}); "
               f"discordant pairs: only-{na} {c.only_a}, only-{nb} {c.only_b}; exact McNemar p = {c.mcnemar_p:.3g}")
    print(md)
    if args.md:
        Path(args.md).write_text(md + "\n")
    if args.plot:
        plot(arms, info, Path(args.plot), args.title)
        print(f"\nplot -> {args.plot}")


if __name__ == "__main__":
    main()
