"""What varies across a task's 50 official init states? Measures settled object poses and the robot
start pose, and overlays per-episode outcomes from evaluation runs.

    uv run python scripts/inspect_init_states.py --suite libero_90 --task-id 56 --object butter_1 \
        --arm B1=results/finetuned/expert_only_s0/libero_90_t56 \
        --arm B2=results/finetuned/expert_only_rehearsal_s0/libero_90_t56
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from vla_sim_poc.config import PROJECT_ROOT, load_config  # noqa: E402
from vla_sim_poc.environment import EpisodeSpec, TaskEnv  # noqa: E402

OK_C, FAIL_C = "#2a78d6", "#e34948"  # categorical blue; red for failures (labelled, not colour-only)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_90")
    ap.add_argument("--task-id", type=int, default=56)
    ap.add_argument("--object", default="butter_1")
    ap.add_argument("--receptacle", default="wooden_tray_1")
    ap.add_argument("--arm", action="append", default=[], help="name=eval dir with episodes.jsonl")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results/plots/t56_init_states.png"))
    args = ap.parse_args()
    env = TaskEnv(args.suite, args.task_id, load_config("eval.yaml")["env"], use_init_states=True)

    rows = []
    for k in range(env.n_init_states):
        raw, _ = env.reset(EpisodeSpec(args.suite, args.task_id, seed=1000, init_state_id=k))
        inner = env.env._env.env
        pose = {}
        for name in inner.objects_dict:
            bid = inner.obj_body_id[name]
            quat_wxyz = inner.sim.data.body_xquat[bid]
            yaw = Rotation.from_quat(np.r_[quat_wxyz[1:], quat_wxyz[0]]).as_euler("xyz", degrees=True)[2]
            pose[name] = (*inner.sim.data.body_xpos[bid][:2].tolist(), float(yaw))
        rows.append({"init": k, "pose": pose, "eef": raw["robot_state"]["eef"]["pos"].tolist(),
                     "joints": raw["robot_state"]["joints"]["pos"].tolist()})
    env.close()

    def spread(vals):
        v = np.asarray(vals)
        return {"min": v.min(0).round(3).tolist(), "max": v.max(0).round(3).tolist(), "std": v.std(0).round(4).tolist()}

    report = {name: spread([r["pose"][name] for r in rows]) for name in rows[0]["pose"]}
    report["robot_eef_xyz"] = spread([r["eef"] for r in rows])
    report["robot_joints"] = spread([r["joints"] for r in rows])
    print(json.dumps(report, indent=1))

    arms = []
    for a in args.arm:
        name, d = a.split("=", 1)
        eps = {json.loads(l)["init_state_id"]: json.loads(l)["success"] for l in (PROJECT_ROOT / d / "episodes.jsonl").read_text().splitlines() if l.strip()}
        arms.append((name, eps))

    fig, axes = plt.subplots(1, max(1, len(arms)), figsize=(3.6 * max(1, len(arms)), 3.6), facecolor=SURFACE, squeeze=False)
    obj = np.array([r["pose"][args.object][:2] for r in rows]) * 100
    for ax, (name, eps) in zip(axes[0], arms or [("init states", {})]):
        ax.set_facecolor(SURFACE)
        ok = np.array([eps.get(r["init"], True) for r in rows])
        ax.scatter(obj[ok, 1], obj[ok, 0], s=28, color=OK_C, edgecolor=SURFACE, lw=1, label="success", zorder=3)
        ax.scatter(obj[~ok, 1], obj[~ok, 0], s=60, color=FAIL_C, marker="X", edgecolor=SURFACE, lw=1, label="failure", zorder=4)
        for r, o, s in zip(rows, obj, ok):
            if not s:
                ax.annotate(str(r["init"]), (o[1], o[0]), xytext=(5, 3), textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(f"{name}: {int(ok.sum())}/{len(ok)}", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("butter y (cm)", fontsize=8, color=INK_2)
        ax.set_ylabel("butter x (cm)", fontsize=8, color=INK_2)
        ax.add_patch(plt.Rectangle((2.5, 2.5), 5, 5, fill=False, ls="--", lw=1, ec=INK_2))  # BDDL region
        ax.set_xlim(1.5, 8.5)
        ax.set_ylim(1.5, 8.5)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=1)
        ax.tick_params(colors=INK_2, length=0, labelsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0][-1].legend(frameon=False, fontsize=7, loc="upper right", labelcolor=INK)
    fig.suptitle("Butter start position in the 50 test states (dashed: 5x5 cm sampling region)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    (Path(args.out).with_suffix(".json")).write_text(json.dumps({"spread": report, "per_state": rows}, indent=1))
    print(f"plot -> {args.out}")


if __name__ == "__main__":
    main()
