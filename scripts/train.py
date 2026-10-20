"""M3: fine-tune SmolVLA on the target-task demos with an explicit, minimal training loop.

Same components as ``lerobot-train`` (policy class, saved pre/post-processors, SmolVLA optimizer and
scheduler presets, LeRobotDataset with 50-step action chunks), with two deliberate differences:
  * the checkpoint's normalization stats are kept (lerobot-train swaps in the dataset's);
  * validation loss on held-out *episodes* is logged with fixed noise so it is comparable over time.

    uv run python scripts/train.py --config configs/train_expert_only.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors.torch import load_file

from vla_sim_poc.config import PROJECT_ROOT, load_config
from vla_sim_poc.manifest import build_manifest, redact_home
from vla_sim_poc.policy import resolve_checkpoint, saved_rename_map
from vla_sim_poc.training import count_params, fixed_noise_loss, make_dataset

NORMALIZER_FILE = "policy_preprocessor_step_5_normalizer_processor.safetensors"


def cycle(loader):
    while True:
        yield from loader


def assert_checkpoint_stats(preprocessor, ckpt_dir: Path) -> None:
    """The training preprocessor must normalize with exactly the checkpoint's stats."""
    ref = load_file(ckpt_dir / NORMALIZER_FILE)
    step = next(s for s in preprocessor.steps if type(s).__name__ == "NormalizerProcessorStep")
    for key in ("observation.state", "action"):
        for stat in ("mean", "std"):
            got = torch.as_tensor(step.stats[key][stat]).float().cpu()
            assert torch.allclose(got, ref[f"{key}.{stat}"].float(), atol=1e-6), f"{key}.{stat} differs from checkpoint"


def save(policy, pre, post, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tr = cfg["training"]
    steps = args.steps or tr["steps"]
    out = PROJECT_ROOT / (args.output_dir or tr["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])

    # --- policy: start from the frozen baseline checkpoint
    ck = cfg["init_checkpoint"]
    ckpt_dir = resolve_checkpoint(ck["repo_id"], ck["revision"])
    assert saved_rename_map(ckpt_dir) == ck["rename_map"]
    pcfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    pcfg.device = "cuda"
    pcfg.freeze_vision_encoder = cfg["trainable"]["freeze_vision_encoder"]
    pcfg.train_expert_only = cfg["trainable"]["train_expert_only"]
    pcfg.pretrained_path = str(ckpt_dir)
    policy = SmolVLAPolicy.from_pretrained(ckpt_dir, config=pcfg).to("cuda")
    policy.train()

    # --- processors: loaded from the checkpoint, NOT rebuilt from dataset stats
    if cfg["normalization"] != "checkpoint":
        raise NotImplementedError("only normalization: checkpoint is supported")
    pre, post = make_pre_post_processors(pcfg, pretrained_path=str(ckpt_dir), preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "rename_observations_processor": {"rename_map": ck["rename_map"]},
    })
    assert_checkpoint_stats(pre, ckpt_dir)

    # --- data: episode-level split
    ds_cfg = cfg["dataset"]
    split = json.loads((PROJECT_ROOT / ds_cfg["split"]).read_text())
    root = PROJECT_ROOT / ds_cfg["root"]
    train_ds = make_dataset(ds_cfg["repo_id"], root, split["train_episodes"], pcfg)
    val_ds = make_dataset(ds_cfg["repo_id"], root, split["val_episodes"], pcfg)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=tr["batch_size"], shuffle=True, num_workers=tr["num_workers"],
        drop_last=True, pin_memory=True, persistent_workers=True,
        generator=torch.Generator().manual_seed(tr["seed"]),
    )

    # --- optimizer / scheduler: SmolVLA presets
    opt_cfg, sched_cfg = pcfg.get_optimizer_preset(), pcfg.get_scheduler_preset()
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = opt_cfg.build(params)
    scheduler = sched_cfg.build(optimizer, num_training_steps=steps)

    run = {
        "config": cfg, "steps": steps, "params": count_params(policy),
        "n_train_frames": len(train_ds), "n_val_frames": len(val_ds),
        "optimizer": {k: v for k, v in vars(opt_cfg).items()}, "scheduler": {k: v for k, v in vars(sched_cfg).items()},
        "init_checkpoint": {**ck, "local_dir": redact_home(ckpt_dir)}, "manifest": build_manifest(),
    }
    (out / "run_config.json").write_text(json.dumps(run, indent=1, default=str))
    print(json.dumps({k: run[k] for k in ("steps", "params", "n_train_frames", "n_val_frames")}))

    log = (out / "train_log.jsonl").open("a")
    v0 = fixed_noise_loss(policy, pre, val_ds, n_batches=8, batch_size=16)
    log.write(json.dumps({"step": 0, "val_loss": v0[0], "val_sem": v0[1]}) + "\n")
    print(f"step 0 val_loss {v0[0]:.4f}")

    torch.cuda.reset_peak_memory_stats()
    t_start, it, window = time.perf_counter(), cycle(loader), []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        batch = pre(next(it))
        loss, _ = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, opt_cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        window.append((loss.item(), grad_norm.item(), time.perf_counter() - t0))

        if step % tr["log_every"] == 0:
            l, g, dt = map(np.mean, zip(*window))
            rec = {"step": step, "loss": l, "grad_norm": g, "lr": scheduler.get_last_lr()[0], "step_s": dt,
                   "elapsed_min": (time.perf_counter() - t_start) / 60}
            window = []
            if step % tr["val_every"] == 0:
                rec["val_loss"], rec["val_sem"] = fixed_noise_loss(policy, pre, val_ds, n_batches=8, batch_size=16)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if step in tr["save_steps"] or step == steps:
            save(policy, pre, post, out / "checkpoints" / f"{step:06d}")

    summary = {
        "steps": steps, "wall_time_min": (time.perf_counter() - t_start) / 60,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0), "cloud_cost_usd": 0.0,
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
