"""M2: cross-check our regenerated dataset against an independent third-party conversion of the same
demos (``jesbu1/libero_90_lerobot``, built with OpenVLA's regeneration + RLDS -> LeRobot pipeline).

For each reference episode, find our episode with the *identical* action sequence (tests the no-op
filter and episode boundaries), then compare per-frame state and images. Images are compared both
as-is and rotated 180 deg, so an orientation mismatch shows up as "rotated matches better".

    uv run python scripts/crosscheck_reference_dataset.py
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from vla_sim_poc.config import PROJECT_ROOT
from vla_sim_poc.video import save_png

REF_ROOT = PROJECT_ROOT / "data/reference/jesbu1_libero_90_lerobot"
INSTRUCTION = "pick up the butter and put it in the tray"


def decode(cell) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def load_reference(root: Path) -> list[dict]:
    eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines()]
    out = []
    for e in eps:
        if e["tasks"] != [INSTRUCTION]:
            continue
        i = e["episode_index"]
        df = pd.read_parquet(root / f"data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet")
        out.append({
            "ref_episode": i,
            "actions": np.stack(df["actions"].to_numpy()).astype(np.float32),
            "states": np.stack(df["state"].to_numpy()).astype(np.float32),
            "image": df["image"].to_numpy(),
            "wrist": df["wrist_image"].to_numpy(),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=str(PROJECT_ROOT / "data/lerobot/libero90_t56"))
    ap.add_argument("--frames-per-episode", type=int, default=5)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results/dataset"))
    args = ap.parse_args()
    ours_root, out = Path(args.ours), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ours = LeRobotDataset("local/libero90_t56_butter_tray", root=ours_root)
    ep_col = np.asarray(ours.hf_dataset["episode_index"])
    side = sorted((ours_root / "vla_sim_poc/sim_states").glob("episode_*.npz"))
    our_eps = [np.load(p) for p in side]
    prov = json.loads((ours_root / "vla_sim_poc/provenance.json").read_text())
    demo_of = {r["episode_index"]: r["demo"] for r in prov["demos"] if "episode_index" in r}

    ref = load_reference(REF_ROOT)
    rows, img_rows, example = [], [], None
    used = set()
    for r in ref:
        match = None
        for j, e in enumerate(our_eps):
            if j not in used and e["actions"].shape == r["actions"].shape and np.allclose(e["actions"], r["actions"], atol=1e-6):
                match = j
                break
        row = {"ref_episode": r["ref_episode"], "ref_len": len(r["actions"]), "our_episode": match,
               "our_demo": demo_of.get(match)}
        if match is not None:
            used.add(match)
            e = our_eps[match]
            row["state_max_abs_diff"] = float(np.abs(e["states"] - r["states"]).max())
            row["state_frame0_abs_diff"] = np.abs(e["states"][0] - r["states"][0]).round(5).tolist()
            idxs = np.nonzero(ep_col == match)[0]
            for t in np.linspace(0, len(idxs) - 1, args.frames_per_episode).astype(int):
                item = ours[int(idxs[t])]
                for key, rk in (("observation.images.image", "image"), ("observation.images.image2", "wrist")):
                    mine = (item[key].permute(1, 2, 0).numpy() * 255).round().astype(np.int16)
                    theirs = decode(r[rk][t]).astype(np.int16)
                    img_rows.append({
                        "camera": rk, "mad": float(np.abs(mine - theirs).mean()),
                        "mad_if_rotated_180": float(np.abs(mine - theirs[::-1, ::-1]).mean()),
                    })
                    if example is None and rk == "image" and t > 0:
                        example = np.concatenate([mine, theirs, np.abs(mine - theirs) * 4], axis=1).clip(0, 255).astype(np.uint8)
        rows.append(row)

    matched = [r for r in rows if r["our_episode"] is not None]
    unmatched_ref = [r["ref_episode"] for r in rows if r["our_episode"] is None]
    img = pd.DataFrame(img_rows)
    report = {
        "reference": "jesbu1/libero_90_lerobot@fa5ab6b7 (task 56 episodes)",
        "n_reference_episodes": len(ref), "n_our_episodes": len(our_eps),
        "n_matched_by_identical_actions": len(matched),
        "unmatched_reference_episodes": unmatched_ref,
        "our_episodes_without_reference": sorted(set(range(len(our_eps))) - used),
        "demos_dropped_by_our_replay": [r["demo"] for r in prov["demos"] if not r["replay_success"]],
        "state_max_abs_diff_over_matched": max((r["state_max_abs_diff"] for r in matched), default=None),
        "state_max_abs_diff_median": float(np.median([r["state_max_abs_diff"] for r in matched])) if matched else None,
        "image_mean_abs_diff_0_255": img.groupby("camera")[["mad", "mad_if_rotated_180"]].mean().round(2).to_dict(),
        "episodes": rows,
    }
    if example is not None:
        save_png(example, out / "crosscheck_ours_ref_diffx4.png")
    (out / "crosscheck_reference.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=1))


if __name__ == "__main__":
    main()
