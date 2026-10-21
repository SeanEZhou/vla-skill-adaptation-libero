"""M3: fine-tune SmolVLA on the target-task demos with an explicit, minimal training loop.

Same components as ``lerobot-train`` (policy class, saved pre/post-processors, SmolVLA optimizer and
scheduler presets, LeRobotDataset with 50-step action chunks), with deliberate differences:
  * the checkpoint's normalization stats are kept (lerobot-train swaps in the dataset's);
  * optional **rehearsal**: batches mix target demos with the checkpoint's original training data;
  * validation loss on held-out target *episodes* + a retention loss on a known task, both with
    fixed noise so they are comparable over time;
  * resumable: a full training state is saved every ``checkpoint_every_min`` minutes and on
    SIGTERM / Ctrl+C (first signal: finish the step, save, exit; second signal: stop immediately).

    uv run python scripts/train.py --config configs/train_expert_only_rehearsal.yaml
    kill -TERM $(cat outputs/train/<run>/train.pid)        # graceful stop (saves, then exits)
    uv run python scripts/train.py --config configs/train_expert_only_rehearsal.yaml --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors.torch import load_file

from vla_sim_poc.config import PROJECT_ROOT, load_config
from vla_sim_poc.dataset import episodes_for_task
from vla_sim_poc.manifest import build_manifest, redact_home
from vla_sim_poc.policy import resolve_checkpoint, saved_rename_map
from vla_sim_poc.training import RetryOnDecodeError, count_params, fixed_noise_loss, make_dataset

NORMALIZER_FILE = "policy_preprocessor_step_5_normalizer_processor.safetensors"
RESUME_DIR, STATE_FILE = "resume", "training_state.pt"


class GracefulStop:
    """First SIGINT/SIGTERM requests a stop at the next step boundary; a second one aborts."""

    def __init__(self):
        self.requested = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, _frame):
        if self.requested:
            raise KeyboardInterrupt(f"second signal {signum}: aborting without saving")
        self.requested = True
        print(f"\n[signal {signum}] stop requested: finishing this step, saving a resumable checkpoint, exiting. "
              "Send again to abort immediately.", flush=True)


def assert_checkpoint_stats(preprocessor, ckpt_dir: Path) -> None:
    """The training preprocessor must normalize with exactly the checkpoint's stats."""
    ref = load_file(ckpt_dir / NORMALIZER_FILE)
    step = next(s for s in preprocessor.steps if type(s).__name__ == "NormalizerProcessorStep")
    for key in ("observation.state", "action"):
        for stat in ("mean", "std"):
            got = torch.as_tensor(step.stats[key][stat]).float().cpu()
            assert torch.allclose(got, ref[f"{key}.{stat}"].float(), atol=1e-6), f"{key}.{stat} differs from checkpoint"


def save_model(policy, pre, post, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)


def save_resume(run_dir: Path, policy, pre, post, optimizer, scheduler, step: int, reason: str) -> None:
    """Write the full training state to <run>/resume atomically (tmp dir, then swap)."""
    tmp, final, old = run_dir / f"{RESUME_DIR}.tmp", run_dir / RESUME_DIR, run_dir / f"{RESUME_DIR}.old"
    for d in (tmp, old):
        if d.exists():
            shutil.rmtree(d)
    t0 = time.perf_counter()
    save_model(policy, pre, post, tmp / "pretrained_model")
    torch.save({
        "step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(), "python": random.getstate()},
        "reason": reason, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, tmp / STATE_FILE)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    if old.exists():
        shutil.rmtree(old)
    print(f"[resume checkpoint] step {step} ({reason}) saved in {time.perf_counter() - t0:.0f}s -> {final}", flush=True)


def build_loader(cfg, pcfg, split, seed: int, steps: int):
    ds_cfg, tr = cfg["dataset"], cfg["training"]
    target = make_dataset(ds_cfg["repo_id"], PROJECT_ROOT / ds_cfg["root"], split["train_episodes"], pcfg)
    reh = cfg.get("rehearsal")
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=tr["batch_size"], num_workers=tr["num_workers"], drop_last=True, pin_memory=True,
              persistent_workers=True)
    if not reh:
        return torch.utils.data.DataLoader(target, shuffle=True, generator=g, **kw), {"target_frames": len(target)}
    rehearsal = make_dataset(reh["repo_id"], None, None, pcfg, revision=reh["revision"])
    both = RetryOnDecodeError(torch.utils.data.ConcatDataset([target, rehearsal]), seed=seed)
    # Per-sample weights so that a fraction `target_fraction` of samples (in expectation) are target frames.
    f = reh["target_fraction"]
    w = torch.cat([torch.full((len(target),), f / len(target)), torch.full((len(rehearsal),), (1 - f) / len(rehearsal))])
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=steps * tr["batch_size"], replacement=True, generator=g)
    return torch.utils.data.DataLoader(both, sampler=sampler, **kw), {
        "target_frames": len(target), "rehearsal_frames": len(rehearsal), "target_fraction": f}


def train_loop(c: dict) -> int:
    """Optimisation loop. Updates c["step"] as it goes so a crash handler can save the right step."""
    it, pre, policy, params, opt_cfg = c["it"], c["pre"], c["policy"], c["params"], c["opt_cfg"]
    optimizer, scheduler, tr, steps, stop, log = c["optimizer"], c["scheduler"], c["tr"], c["steps"], c["stop"], c["log"]
    out, post, evaluate_losses, t_start = c["out"], c["post"], c["evaluate_losses"], c["t_start"]
    step, window, last_ckpt = c["step"], [], time.perf_counter()
    while step < steps and not stop.requested:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except Exception:
            if stop.requested:  # Ctrl+C in a terminal also kills the loader workers
                break
            raise
        batch = pre(batch)
        loss, _ = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, opt_cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        c["step"] = step
        if os.environ.get("VLA_DEBUG_CRASH_AT_STEP") == str(step):  # test hook for the crash-save path
            raise RuntimeError(f"debug crash injected at step {step}")
        window.append((loss.item(), grad_norm.item(), time.perf_counter() - t0))

        if step % tr["log_every"] == 0:
            l, g, dt = map(np.mean, zip(*window))
            rec = {"step": step, "loss": l, "grad_norm": g, "lr": scheduler.get_last_lr()[0], "step_s": dt,
                   "elapsed_min": (time.perf_counter() - t_start) / 60}
            window = []
            if step % tr["val_every"] == 0:
                rec.update(evaluate_losses())
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if step in tr["save_steps"] or step == steps:
            save_model(policy, pre, post, out / "checkpoints" / f"{step:06d}")
        if step in tr["save_steps"] or time.perf_counter() - last_ckpt >= tr["checkpoint_every_min"] * 60:
            save_resume(out, policy, pre, post, optimizer, scheduler, step,
                        "eval checkpoint" if step in tr["save_steps"] else "periodic")
            last_ckpt = time.perf_counter()

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from <output_dir>/resume")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tr = cfg["training"]
    steps = args.steps or tr["steps"]
    out = PROJECT_ROOT / (args.output_dir or tr["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.pid").write_text(str(os.getpid()))
    stop = GracefulStop()

    state = None
    if args.resume:
        state = torch.load(out / RESUME_DIR / STATE_FILE, weights_only=False)
        print(f"resuming from step {state['step']} (saved {state['saved_at']}, reason: {state['reason']})")
    start_step = state["step"] if state else 0
    seed = tr["seed"] + start_step  # fresh data order after resume (documented: not bit-identical)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- policy: start from the frozen baseline checkpoint (or the resume weights)
    ck = cfg["init_checkpoint"]
    ckpt_dir = resolve_checkpoint(ck["repo_id"], ck["revision"])
    assert saved_rename_map(ckpt_dir) == ck["rename_map"]
    pcfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    pcfg.device = "cuda"
    pcfg.freeze_vision_encoder = cfg["trainable"]["freeze_vision_encoder"]
    pcfg.train_expert_only = cfg["trainable"]["train_expert_only"]
    pcfg.pretrained_path = str(ckpt_dir)
    weights_dir = out / RESUME_DIR / "pretrained_model" if state else ckpt_dir
    policy = SmolVLAPolicy.from_pretrained(weights_dir, config=pcfg).to("cuda")
    policy.train()

    # --- processors: loaded from the checkpoint, NOT rebuilt from dataset stats
    if cfg["normalization"] != "checkpoint":
        raise NotImplementedError("only normalization: checkpoint is supported")
    pre, post = make_pre_post_processors(pcfg, pretrained_path=str(ckpt_dir), preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "rename_observations_processor": {"rename_map": ck["rename_map"]},
    })
    assert_checkpoint_stats(pre, ckpt_dir)

    # --- data: episode-level split for the target; optional rehearsal mix; monitors
    ds_cfg = cfg["dataset"]
    split = json.loads((PROJECT_ROOT / ds_cfg["split"]).read_text())
    loader, data_info = build_loader(cfg, pcfg, split, seed, steps - start_step)
    val_ds = make_dataset(ds_cfg["repo_id"], PROJECT_ROOT / ds_cfg["root"], split["val_episodes"], pcfg)
    mon = cfg.get("retention_monitor")
    ret_ds = None
    if mon:
        eps = episodes_for_task(mon["repo_id"], mon["revision"], mon["task_index"])[: mon["n_episodes"]]
        ret_ds = make_dataset(mon["repo_id"], None, eps, pcfg, revision=mon["revision"])

    # --- optimizer / scheduler: SmolVLA presets
    opt_cfg, sched_cfg = pcfg.get_optimizer_preset(), pcfg.get_scheduler_preset()
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = opt_cfg.build(params)
    scheduler = sched_cfg.build(optimizer, num_training_steps=steps)
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])

    run = {
        "config": cfg, "steps": steps, "params": count_params(policy), "data": data_info,
        "n_val_frames": len(val_ds), "optimizer": vars(opt_cfg), "scheduler": vars(sched_cfg),
        "init_checkpoint": {**ck, "local_dir": redact_home(ckpt_dir)}, "manifest": build_manifest(),
    }
    (out / ("run_config.json" if not state else f"run_config_resume_{start_step:06d}.json")).write_text(
        json.dumps(run, indent=1, default=str))
    print(json.dumps({k: run[k] for k in ("steps", "params", "data", "n_val_frames")}))

    def evaluate_losses() -> dict:
        rec = {}
        rec["val_loss"], rec["val_sem"] = fixed_noise_loss(policy, pre, val_ds, n_batches=8, batch_size=16)
        if ret_ds is not None:
            rec["retention_loss"], rec["retention_sem"] = fixed_noise_loss(policy, pre, ret_ds, n_batches=8, batch_size=16)
        return rec

    log = (out / "train_log.jsonl").open("a")
    if not state:
        rec = {"step": 0, **evaluate_losses()}
        log.write(json.dumps(rec) + "\n")
        print(rec, flush=True)

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    ctx = dict(it=iter(loader), pre=pre, policy=policy, params=params, opt_cfg=opt_cfg, optimizer=optimizer,
               scheduler=scheduler, tr=tr, steps=steps, step=start_step, stop=stop, log=log, out=out, post=post,
               evaluate_losses=evaluate_losses, t_start=t_start)
    try:
        step = train_loop(ctx)
    except KeyboardInterrupt:
        raise  # second signal: abort without saving, as documented
    except BaseException:
        # Crash safety: parameters and optimizer are consistent between steps, so save before dying.
        save_resume(out, policy, pre, post, optimizer, scheduler, ctx["step"], "crash")
        raise

    if step < steps:
        save_resume(out, policy, pre, post, optimizer, scheduler, step, "stop requested")
        print(f"stopped at step {step}/{steps}; resume with --resume")
    else:
        save_resume(out, policy, pre, post, optimizer, scheduler, step, "finished")
    summary = {
        "steps_done": step, "steps_total": steps, "segment_start_step": start_step,
        "segment_wall_time_min": (time.perf_counter() - t_start) / 60,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0), "cloud_cost_usd": 0.0,
    }
    with (out / "train_segments.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    if step == steps:
        (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
