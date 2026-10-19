"""Closed-loop evaluation of one policy on explicit LIBERO initial conditions.

Each episode is either an official init state (``--init-states 0-49``) or a seeded sampled layout
(``--seeds 10000-10009``). Results append to ``<out>/episodes.jsonl``; re-running resumes and skips
finished episodes. Every episode gets a video in ``<out>/videos``.

    # final evaluation on frozen init states
    uv run python scripts/evaluate.py --suite libero_90 --task-ids 57 --init-states 0-49 \
        --out results/baseline/libero_90_t57
    # screening on sampled layouts
    uv run python scripts/evaluate.py --suite libero_90 --task-ids 46,57 --seeds 10000-10009 \
        --out results/screening/libero_90
"""

from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np

from vla_sim_poc.config import PROJECT_ROOT, load_config, parse_id_list
from vla_sim_poc.environment import EpisodeSpec, TaskEnv
from vla_sim_poc.manifest import build_manifest
from vla_sim_poc.metrics import proportion
from vla_sim_poc.policy import load_policy
from vla_sim_poc.rollout import run_episode
from vla_sim_poc.video import write_mp4


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["eval.yaml", "baseline.yaml"])
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task-ids", required=True, help='e.g. "0" or "0-9,46"')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--init-states", help='official init state ids, e.g. "0-49"')
    g.add_argument("--seeds", help='seeded sampled layouts, e.g. "10000-10009"')
    ap.add_argument("--seed", type=int, default=1000, help="env seed used with --init-states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="override the per-suite episode limit")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. policy.n_action_steps=50")
    args = ap.parse_args()

    cfg = load_config(*args.configs)
    for kv in args.set:
        dotted, val = kv.split("=", 1)
        *parents, leaf = dotted.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        node[leaf] = type(node[leaf])(val) if node.get(leaf) is not None else val

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "actions").mkdir(parents=True, exist_ok=True)
    ep_path = out / "episodes.jsonl"
    done = load_done(ep_path)

    bundle = load_policy(cfg["checkpoint"], cfg["policy"])
    run_info = {"args": vars(args), "config": cfg, "policy_source": bundle.source, "manifest": build_manifest()}
    (out / f"run_config_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json").write_text(json.dumps(run_info, indent=1))
    name = cfg.get("name", "policy")
    use_init = args.init_states is not None
    ids = parse_id_list(args.init_states if use_init else args.seeds)

    for task_id in parse_id_list(args.task_ids):
        env = TaskEnv(args.suite, task_id, cfg["env"], use_init_states=use_init)
        print(f"\n=== {args.suite} task {task_id}: {env.task_description!r} (max {env.max_steps} steps)")
        for i in ids:
            spec = EpisodeSpec(args.suite, task_id, seed=args.seed if use_init else i, init_state_id=i if use_init else None)
            if spec.key in done:
                continue
            result, frames, actions = run_episode(
                env, spec, bundle, name, cfg["env"]["image_size"], max_steps=args.max_steps, record_frames=not args.no_video
            )
            if not args.no_video:
                vpath = out / "videos" / f"{spec.key}_{'success' if result.success else 'fail'}.mp4"
                write_mp4(frames, vpath, fps=cfg["video"]["fps"])
                result.video_path = str(vpath.relative_to(PROJECT_ROOT))
            # Raw (pre-clip) policy actions, for action-distribution analysis of failures.
            np.save(out / "actions" / f"{spec.key}.npy", actions.astype(np.float32))
            with ep_path.open("a") as f:
                f.write(json.dumps(result.to_json()) + "\n")
            done[spec.key] = result.to_json()
            print(
                f"  {spec.key}: {'SUCCESS' if result.success else 'fail   '} steps={result.steps:3d} "
                f"wall={result.wall_time_s:5.1f}s infer={result.inference_latency_mean_s * 1e3:5.0f}ms "
                f"grip_close={result.action_stats.get('gripper_close_frac', 0):.2f}"
            )
        env.close()

    # Summary over every episode in this output dir, per task.
    rows = list(load_done(ep_path).values())
    summary = {"policy": name, "per_task": {}}
    for tid in sorted({r["task_id"] for r in rows}):
        rs = [r for r in rows if r["task_id"] == tid]
        p = proportion([r["success"] for r in rs])
        summary["per_task"][tid] = {
            "instruction": rs[0]["instruction"],
            "successes": p.successes, "n": p.n, "rate": p.rate, "ci95": [p.ci_low, p.ci_high],
            "mean_steps": sum(r["steps"] for r in rs) / len(rs),
            "eval_wall_time_s": sum(r["wall_time_s"] for r in rs),
            "inference_latency_mean_ms": 1e3 * sum(r["inference_latency_mean_s"] for r in rs) / len(rs),
        }
        print(f"[task {tid}] {p}  {rs[0]['instruction']}")
    p = proportion([r["success"] for r in rows])
    summary["overall"] = {"successes": p.successes, "n": p.n, "rate": p.rate, "ci95": [p.ci_low, p.ci_high]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[overall] {p}")


if __name__ == "__main__":
    main()
