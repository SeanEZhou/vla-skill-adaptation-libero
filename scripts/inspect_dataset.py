"""Inspect a LeRobot-format LIBERO dataset and cross-check it against the live simulator.

Checks (per task):
* schema: feature keys / shapes / dtypes
* frame-0 ``observation.state`` of demo episodes vs the env's post-reset state on the official
  init states (same frame, same units -> distributions should overlap closely)
* frame-0 camera images vs the env's policy-input images (same orientation / camera assignment)
* action ranges and gripper convention (-1 open / +1 close)

    uv run python scripts/inspect_dataset.py --suite libero_object --task-id 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_sim_poc.config import PROJECT_ROOT, load_config
from vla_sim_poc.environment import EpisodeSpec, TaskEnv
from vla_sim_poc.observation_adapter import make_env_preprocessor, raw_to_policy_frame
from vla_sim_poc.video import save_png


def to_hwc_uint8(chw) -> np.ndarray:
    return (chw.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def episodes_for_task(repo_id: str, revision: str, task_index: int) -> list[int]:
    """Episode -> task mapping from the (small) data parquet files; episode metadata lacks it."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id, repo_type="dataset", revision=revision, allow_patterns=["data/*"]))
    df = pd.concat(pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in sorted(root.glob("data/*/*.parquet")))
    per_ep = df.groupby("episode_index")["task_index"].agg(["min", "max"])
    assert (per_ep["min"] == per_ep["max"]).all(), "an episode spans several tasks"
    return sorted(int(e) for e in per_ep.index[per_ep["min"] == task_index])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="lerobot/libero")
    ap.add_argument("--revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--n-env-states", type=int, default=10)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results/smoke"))
    args = ap.parse_args()
    out = Path(args.out)
    cfg = load_config("eval.yaml")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    env = TaskEnv(args.suite, args.task_id, cfg["env"], use_init_states=True)
    instruction = env.task_description

    meta = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)
    tasks = meta.tasks  # DataFrame indexed by task string
    if instruction not in tasks.index:
        raise SystemExit(f"Instruction {instruction!r} not in dataset tasks:\n{list(tasks.index)}")
    task_index = int(tasks.loc[instruction, "task_index"])
    ep_ids = episodes_for_task(args.repo_id, args.revision, task_index)
    print(f"task_index={task_index}  episodes for this instruction: {len(ep_ids)}")
    sel = ep_ids[: args.n_episodes]
    ds = LeRobotDataset(args.repo_id, revision=args.revision, episodes=sel)

    ep_from = {int(e): None for e in sel}
    states0, actions, lengths = [], [], []
    first_img = None
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    for e in sel:
        idxs = np.nonzero(ep_col == e)[0]
        lengths.append(len(idxs))
        item0 = ds[int(idxs[0])]
        states0.append(item0["observation.state"].numpy())
        actions.append(np.asarray(ds.hf_dataset.select(idxs.tolist())["action"]))
        if first_img is None:
            first_img = np.concatenate(
                [to_hwc_uint8(item0["observation.images.image"]), to_hwc_uint8(item0["observation.images.image2"])], axis=1
            )
        ep_from[e] = int(idxs[0])
    states0 = np.stack(states0)
    acts = np.concatenate(actions)

    env_states, env_img = [], None
    pre = make_env_preprocessor()
    for i in range(args.n_env_states):
        raw, _ = env.reset(EpisodeSpec(args.suite, args.task_id, seed=0, init_state_id=i))
        fr = raw_to_policy_frame(raw, instruction, pre)
        env_states.append(fr["observation.state"][0].numpy())
        if env_img is None:
            env_img = np.concatenate(
                [to_hwc_uint8(fr["observation.images.image"][0]), to_hwc_uint8(fr["observation.images.image2"][0])], axis=1
            )
    env.close()
    env_states = np.stack(env_states)

    save_png(np.concatenate([first_img, env_img], axis=0), out / f"dataset_vs_env_{args.suite}_t{args.task_id}.png")
    report = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "suite": args.suite,
        "task_id": args.task_id,
        "instruction": instruction,
        "n_episodes_for_task": len(ep_ids),
        "episode_lengths": lengths,
        "features": {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in meta.features.items()},
        "fps_in_metadata": meta.fps,
        "frame0_state_dataset_mean": states0.mean(0).round(4).tolist(),
        "frame0_state_env_mean": env_states.mean(0).round(4).tolist(),
        "frame0_state_dataset_std": states0.std(0).round(4).tolist(),
        "frame0_state_env_std": env_states.std(0).round(4).tolist(),
        "frame0_state_mean_abs_diff": np.abs(states0.mean(0) - env_states.mean(0)).round(4).tolist(),
        "action_min": acts.min(0).round(3).tolist(),
        "action_max": acts.max(0).round(3).tolist(),
        "gripper_values": sorted(set(np.round(acts[:, 6], 3).tolist()))[:10],
        "gripper_first_action_mean": float(np.mean([a[0, 6] for a in actions])),
    }
    (out / f"dataset_check_{args.suite}_t{args.task_id}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
