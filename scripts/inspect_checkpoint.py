"""M0: print the checkpoint's expected feature schema, normalization stats and inference settings,
and compare its stats against the training dataset's published stats.

    uv run python scripts/inspect_checkpoint.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vla_sim_poc.config import PROJECT_ROOT, load_config
from vla_sim_poc.manifest import redact_home
from vla_sim_poc.policy import resolve_checkpoint, saved_rename_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline.yaml")
    ap.add_argument("--dataset", default="lerobot/libero")
    ap.add_argument("--dataset-revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results/smoke/checkpoint_schema.json"))
    args = ap.parse_args()
    ck = load_config(args.config)["checkpoint"]
    d = resolve_checkpoint(ck["repo_id"], ck["revision"])
    cfg = json.loads((d / "config.json").read_text())
    train_cfg = json.loads((d / "train_config.json").read_text()) if (d / "train_config.json").exists() else {}

    pre = load_file(d / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    post = load_file(d / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    stats = {k: v.numpy().round(5).tolist() for k, v in pre.items() if k.split(".")[-1] in ("mean", "std")}
    post_stats = {k: v.numpy().round(5).tolist() for k, v in post.items() if k.split(".")[-1] in ("mean", "std")}

    ds_stats = json.loads(
        Path(hf_hub_download(args.dataset, "meta/stats.json", repo_type="dataset", revision=args.dataset_revision)).read_text()
    )
    comparison = {}
    for feat in ("observation.state", "action"):
        for s in ("mean", "std"):
            ck_v = np.asarray(pre[f"{feat}.{s}"].numpy())
            ds_v = np.asarray(ds_stats[feat][s])
            comparison[f"{feat}.{s}"] = {"max_abs_diff_vs_dataset": float(np.abs(ck_v - ds_v).max())}

    report = {
        "checkpoint": {**ck, "local_dir": redact_home(d)},
        "type": cfg["type"],
        "input_features": cfg["input_features"],
        "output_features": cfg["output_features"],
        "chunk_size": cfg["chunk_size"],
        "n_action_steps_shipped": cfg["n_action_steps"],
        "num_steps": cfg["num_steps"],
        "resize_imgs_with_padding": cfg["resize_imgs_with_padding"],
        "max_state_dim": cfg["max_state_dim"],
        "max_action_dim": cfg["max_action_dim"],
        "normalization_mapping": cfg["normalization_mapping"],
        "vlm_model_name": cfg["vlm_model_name"],
        "num_vlm_layers": cfg["num_vlm_layers"],
        "expert_width_multiplier": cfg["expert_width_multiplier"],
        "pretrained_from": cfg.get("pretrained_path"),
        "saved_rename_map": saved_rename_map(d),
        "train_config": {
            "dataset": train_cfg.get("dataset", {}).get("repo_id"),
            "dataset_revision": train_cfg.get("dataset", {}).get("revision"),
            "steps": train_cfg.get("steps"),
            "batch_size": train_cfg.get("batch_size"),
            "env_fps_in_train_config": train_cfg.get("env", {}).get("fps"),
        },
        "normalizer_stats": stats,
        "unnormalizer_stats_equal_normalizer": all(
            np.allclose(post_stats[k], stats[k]) for k in post_stats if k in stats
        ),
        "stats_vs_dataset": comparison,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
