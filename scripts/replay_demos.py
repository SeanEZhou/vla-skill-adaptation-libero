"""Validate the simulator's success detector with known-good expert trajectories, and check whether
demo initial states coincide with the official evaluation init states (train/eval leakage).

For each original LIBERO HDF5 demo: restore its first MuJoCo state, replay its recorded actions
open-loop, and record whether / when ``check_success()`` fires. Negative control: the same start
state with no-op actions must not succeed.

    uv run python scripts/replay_demos.py --suite libero_object --task-id 0 \
        --hdf5 libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import hf_hub_download

from vla_sim_poc.config import PROJECT_ROOT, load_config
from vla_sim_poc.environment import TaskEnv
from vla_sim_poc.video import viewer_frame, write_mp4

HDF5_REPO = "yifengzhu-hf/LIBERO-datasets"
HDF5_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
NOOP = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def replay(env: TaskEnv, state0: np.ndarray, actions: np.ndarray, frames: list | None = None):
    inner = env.env._env
    inner.reset()
    raw = inner.set_init_state(state0)
    first_success = None
    for t, a in enumerate(actions):
        raw, _, _, _ = inner.step(a)
        if frames is not None:
            frames.append(viewer_frame(env.env._format_raw_obs(raw)))
        if first_success is None and inner.check_success():
            first_success = t + 1
    return first_success, bool(inner.check_success())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--hdf5", required=True, help="path inside the HF repo, e.g. libero_90/<TASK>_demo.hdf5")
    ap.add_argument("--n-demos", type=int, default=50)
    ap.add_argument("--n-videos", type=int, default=2)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results/smoke/replay"))
    args = ap.parse_args()
    cfg = load_config("eval.yaml")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(HDF5_REPO, args.hdf5, repo_type="dataset", revision=HDF5_REVISION,
                           local_dir=PROJECT_ROOT / "data/libero_hdf5")
    env = TaskEnv(args.suite, args.task_id, cfg["env"], use_init_states=True)
    env.env._ensure_env()
    official = np.asarray(env.env._init_states)
    # Flattened MuJoCo state = [time, qpos(nq), qvel(nv), ...]. Compare object/robot *positions* only:
    # official init states are settled snapshots (zero velocity, time 0), demo starts are mid-settle.
    nq = env.sim.model.nq
    qpos = slice(1, 1 + nq)

    rows = []
    with h5py.File(path) as f:
        data = f["data"]
        h5_instr = json.loads(data.attrs["problem_info"])["language_instruction"]
        if h5_instr != env.task_description:
            raise SystemExit(f"Instruction mismatch: hdf5 {h5_instr!r} vs env {env.task_description!r}")
        demos = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[: args.n_demos]
        for i, k in enumerate(demos):
            states, actions = data[k]["states"][()], data[k]["actions"][()]
            frames = [] if i < args.n_videos else None
            first, final = replay(env, states[0], actions, frames)
            if frames:
                write_mp4(frames, out / f"{args.suite}_t{args.task_id:02d}_{k}_{'success' if final else 'fail'}.mp4")
            # Leakage: distance from this demo's start state to the nearest official init state.
            d = np.linalg.norm(official[:, qpos] - states[0][None, qpos], axis=1) if states.shape[1] == official.shape[1] else None
            rows.append({
                "demo": k, "len": len(actions), "first_success_step": first, "success_at_end": final,
                "nearest_official_init_dist": float(d.min()) if d is not None else None,
                "nearest_official_init_id": int(d.argmin()) if d is not None else None,
            })
            print(rows[-1])
        # Negative control on the first demo's start state.
        s0 = data[demos[0]]["states"][0]
        neg_first, neg_final = replay(env, s0, np.repeat(NOOP[None], len(data[demos[0]]["actions"]), 0))
    env.close()

    # Scale reference: nearest-neighbour distances among the official init states themselves.
    dd = np.linalg.norm(official[:, None, qpos] - official[None, :, qpos], axis=-1)
    np.fill_diagonal(dd, np.inf)
    dists = [r["nearest_official_init_dist"] for r in rows if r["nearest_official_init_dist"] is not None]
    report = {
        "suite": args.suite, "task_id": args.task_id, "hdf5": args.hdf5, "hdf5_revision": HDF5_REVISION,
        "n_demos": len(rows),
        "replay_success_rate": float(np.mean([r["success_at_end"] for r in rows])),
        "replay_first_success_step_median": float(np.median([r["first_success_step"] for r in rows if r["first_success_step"]] or [np.nan])),
        "noop_negative_control_success": neg_final or bool(neg_first),
        "leakage": {
            "compared": f"qpos only (nq={nq})",
            "state_dim_demo": int(s0.shape[0]), "state_dim_official": int(official.shape[1]),
            "min_demo_to_official_dist": float(min(dists)) if dists else None,
            "median_demo_to_official_dist": float(np.median(dists)) if dists else None,
            "median_official_nn_dist": float(np.median(dd.min(axis=1))),
            "n_demos_within_1e-6": int(sum(d < 1e-6 for d in dists)),
        },
        "demos": rows,
    }
    (out / f"replay_{args.suite}_t{args.task_id:02d}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "demos"}, indent=1))


if __name__ == "__main__":
    main()
