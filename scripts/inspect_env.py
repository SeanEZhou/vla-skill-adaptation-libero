"""M0: reset / render / step a LIBERO task, print the actual observation & action schemas, save PNGs.

    uv run python scripts/inspect_env.py --suite libero_object --task-id 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_sim_poc.config import PROJECT_ROOT, load_config
from vla_sim_poc.environment import EpisodeSpec, TaskEnv
from vla_sim_poc.observation_adapter import make_env_preprocessor, raw_to_policy_frame, validate_frame
from vla_sim_poc.video import save_png, viewer_frame


def describe(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(describe(v, f"{prefix}{k}/"))
    elif hasattr(x, "shape"):
        arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        info = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.size and np.issubdtype(arr.dtype, np.number):
            info.update(min=float(arr.min()), max=float(arr.max()))
        out[prefix.rstrip("/")] = info
    else:
        out[prefix.rstrip("/")] = repr(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results/smoke"))
    args = ap.parse_args()
    cfg = load_config("eval.yaml")
    out = Path(args.out)

    env = TaskEnv(args.suite, args.task_id, cfg["env"], use_init_states=True)
    spec = EpisodeSpec(args.suite, args.task_id, seed=0, init_state_id=args.init_state_id)
    raw, info = env.reset(spec)
    frame = raw_to_policy_frame(raw, env.task_description, make_env_preprocessor())
    validate_frame(frame, ["observation.images.image", "observation.images.image2"], cfg["env"]["image_size"])

    ctrl = env.env._env.robots[0].controller
    report = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": env.task_name,
        "instruction": env.task_description,
        "n_init_states": env.n_init_states,
        "max_steps": env.max_steps,
        "raw_observation": describe(raw),
        "policy_frame_before_policy_preprocessor": describe({k: v for k, v in frame.items() if k != "task"}),
        "action_space": {
            "shape": list(env.env.action_space.shape),
            "low": env.env.action_space.low.tolist(),
            "high": env.env.action_space.high.tolist(),
        },
        "controller": {
            "name": type(ctrl).__name__,
            "use_delta": bool(ctrl.use_delta),
            "control_freq": env.env._env.env.control_freq,
            "input_min": np.asarray(ctrl.input_min).tolist(),
            "input_max": np.asarray(ctrl.input_max).tolist(),
            "output_min": np.asarray(ctrl.output_min).tolist(),
            "output_max": np.asarray(ctrl.output_max).tolist(),
            "control_dim": int(ctrl.control_dim),
        },
    }

    # Images: raw render (upside down), and what the policy receives (rotated upright).
    save_png(np.concatenate([raw["pixels"]["image"], raw["pixels"]["image2"]], axis=1), out / "render_raw.png")
    pol = [
        (frame[k][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        for k in ("observation.images.image", "observation.images.image2")
    ]
    save_png(np.concatenate(pol, axis=1), out / "render_policy_input.png")

    # Step: no-op should neither move objects nor trigger success; a +z action should raise the EEF.
    z0 = raw["robot_state"]["eef"]["pos"].copy()
    for _ in range(10):
        raw, _, _, _, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
    z_noop = raw["robot_state"]["eef"]["pos"].copy()
    for _ in range(10):
        raw, _, _, _, info = env.step(np.array([0, 0, 1, 0, 0, 0, -1], dtype=np.float32))
    z_up = raw["robot_state"]["eef"]["pos"].copy()
    save_png(viewer_frame(raw), out / "render_after_up.png")
    report["step_checks"] = {
        "eef_drift_after_10_noops_m": float(np.linalg.norm(z_noop - z0)),
        "eef_dz_after_10_up_actions_m": float(z_up[2] - z_noop[2]),
        "expected_dz_upper_bound_m": 10 * 0.05,
        "success_after_steps": bool(info["is_success"]),
    }
    env.close()

    out.mkdir(parents=True, exist_ok=True)
    (out / "env_schema.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"\nSaved {out}/render_raw.png, render_policy_input.png, render_after_up.png, env_schema.json")


if __name__ == "__main__":
    main()
