"""M2: build the target-task LeRobot dataset from the official LIBERO HDF5 demos.

Outputs (under --out, git-ignored):
    <out>/                      LeRobot v3 dataset (images lossless PNG, fps=20)
    <out>/vla_sim_poc/provenance.json   source file + revision, per-demo replay outcome, no-op counts
    <out>/vla_sim_poc/split.json        episode-level train/val split
    <out>/vla_sim_poc/sim_states/       per-episode MuJoCo states + actions (for alignment tests)
    <out>/vla_sim_poc/stats_vs_checkpoint.json   normalization shift vs the checkpoint's stats

    uv run python scripts/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vla_sim_poc.config import PROJECT_ROOT, load_config
from vla_sim_poc.dataset import (
    ACTION_KEY,
    IMAGE_KEYS,
    STATE_KEY,
    compute_stats,
    episode_split,
    lerobot_features,
    regenerate_episode,
    stats_shift_in_std,
)
from vla_sim_poc.environment import TaskEnv
from vla_sim_poc.manifest import build_manifest, redact_home
from vla_sim_poc.policy import resolve_checkpoint

HDF5_REPO = "yifengzhu-hf/LIBERO-datasets"
HDF5_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["eval.yaml", "baseline.yaml", "selected_task.yaml"])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data/lerobot/libero90_t56"))
    ap.add_argument("--repo-id", default="local/libero90_t56_butter_tray")
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(*args.configs)
    task, out = cfg["task"], Path(args.out)

    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to rebuild")
        if out.resolve().parent != (PROJECT_ROOT / "data/lerobot").resolve():
            raise SystemExit(f"Refusing to delete {out}: not directly under data/lerobot/")
        shutil.rmtree(out)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    h5_path = hf_hub_download(HDF5_REPO, cfg["data"]["hdf5"], repo_type="dataset", revision=HDF5_REVISION,
                              local_dir=PROJECT_ROOT / "data/libero_hdf5")
    env = TaskEnv(task["suite"], task["task_id"], cfg["env"], use_init_states=True)
    if env.task_description != task["instruction"]:
        raise SystemExit(f"Env instruction {env.task_description!r} != config {task['instruction']!r}")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=cfg["env"]["control_freq"], features=lerobot_features(cfg["env"]["image_size"]),
        root=out, robot_type="panda", use_videos=False, image_writer_threads=4,
    )
    side = out / "vla_sim_poc"
    (side / "sim_states").mkdir(parents=True, exist_ok=True)

    rows, kept, all_states, all_actions = [], [], [], []
    with h5py.File(h5_path) as f:
        data = f["data"]
        h5_instr = json.loads(data.attrs["problem_info"])["language_instruction"]
        if h5_instr != env.task_description:
            raise SystemExit(f"HDF5 instruction {h5_instr!r} != env {env.task_description!r}")
        demos = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[: args.max_demos]
        for k in demos:
            ep = regenerate_episode(env, k, data[k]["states"][0], data[k]["actions"][()])
            row = {"demo": k, "orig_len": int(len(data[k]["actions"])), "kept_len": len(ep), "n_noops": ep.n_noops,
                   "replay_success": ep.success, "first_success_frame": ep.first_success_frame}
            if ep.success:
                ep_idx = len(kept)
                for t in range(len(ep)):
                    ds.add_frame({
                        IMAGE_KEYS[0]: ep.images[IMAGE_KEYS[0]][t],
                        IMAGE_KEYS[1]: ep.images[IMAGE_KEYS[1]][t],
                        STATE_KEY: ep.states[t],
                        ACTION_KEY: ep.actions[t],
                        "task": env.task_description,
                    })
                ds.save_episode()
                np.savez_compressed(side / "sim_states" / f"episode_{ep_idx:03d}.npz",
                                    sim_states=np.stack(ep.sim_states), actions=np.stack(ep.actions),
                                    states=np.stack(ep.states))
                row["episode_index"] = ep_idx
                kept.append(k)
                all_states.append(np.stack(ep.states))
                all_actions.append(np.stack(ep.actions))
            rows.append(row)
            print(row)
    ds.finalize()
    env.close()

    train, val = episode_split(len(kept), args.n_val, args.split_seed)
    (side / "split.json").write_text(json.dumps({
        "seed": args.split_seed, "train_episodes": train, "val_episodes": val,
        "note": "episode-level split; val used only for offline loss, never for closed-loop eval",
    }, indent=1))

    # Normalization: the new data's stats vs the stats the checkpoint was trained with.
    ck = load_file(resolve_checkpoint(cfg["checkpoint"]["repo_id"], cfg["checkpoint"]["revision"])
                   / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    shift = {}
    for key, arr in ((STATE_KEY, np.concatenate(all_states)), (ACTION_KEY, np.concatenate(all_actions))):
        new = compute_stats(arr)
        ref = {s: ck[f"{key}.{s}"].numpy() for s in ("mean", "std")}
        shift[key] = {
            "new_mean": new["mean"].round(4).tolist(), "new_std": new["std"].round(4).tolist(),
            "ckpt_mean": ref["mean"].round(4).tolist(), "ckpt_std": ref["std"].round(4).tolist(),
            "mean_shift_in_ckpt_std": stats_shift_in_std(new, ref).round(2).tolist(),
            "std_ratio_new_over_ckpt": (new["std"] / (ref["std"] + 1e-8)).round(2).tolist(),
        }
    (side / "stats_vs_checkpoint.json").write_text(json.dumps(shift, indent=1))

    provenance = {
        "source": {"repo": HDF5_REPO, "revision": HDF5_REVISION, "file": cfg["data"]["hdf5"]},
        "task": task, "instruction": env.task_description, "fps": cfg["env"]["control_freq"],
        "image_size": cfg["env"]["image_size"], "image_storage": "png (lossless)",
        "recipe": "reset -> set_init_state(demo.states[0]) -> 10 settle no-ops -> skip no-op actions -> "
                  "record obs then step; keep only successful replays (OpenVLA regenerate_libero_dataset.py)",
        "n_demos": len(demos), "n_kept": len(kept), "n_frames": int(sum(len(a) for a in all_actions)),
        "demos": rows, "manifest": build_manifest(), "out": redact_home(out),
    }
    (side / "provenance.json").write_text(json.dumps(provenance, indent=1))
    print(f"\nkept {len(kept)}/{len(demos)} demos, {provenance['n_frames']} frames -> {out}")
    print(f"split: {len(train)} train / {len(val)} val episodes; val={val}")
    for key, v in shift.items():
        print(f"{key}: mean shift (ckpt std units) {v['mean_shift_in_ckpt_std']}  std ratio {v['std_ratio_new_over_ckpt']}")


if __name__ == "__main__":
    main()
