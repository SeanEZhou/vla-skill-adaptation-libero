"""Policy output -> LIBERO controller input.

Action semantics (verified in robosuite 1.4.0 ``osc_pose.json`` + LIBERO ``env_wrapper.py``):

    a = [dx, dy, dz, dax, day, daz, gripper]  in [-1, 1]

* Controller: OSC_POSE, ``control_delta=True`` (``control_mode=relative``), 20 Hz.
* Dims 0-2: end-effector position delta in the robot base frame, scaled by ``output_max=0.05`` m.
* Dims 3-5: end-effector orientation delta (axis-angle), scaled by ``output_max=0.5`` rad.
* Dim 6: gripper, -1 = open, +1 = close.

The policy postprocessor has already un-normalized the action (MEAN_STD with the checkpoint's
dataset stats), so values arrive in controller units. robosuite clips to [-1, 1] internally; we clip
explicitly too so the clip rate is measurable rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ACTION_DIM = 7
ACTION_LOW, ACTION_HIGH = -1.0, 1.0
ACTION_LAYOUT = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")
POS_SCALE_M = 0.05
ROT_SCALE_RAD = 0.5


@dataclass
class EnvAction:
    action: np.ndarray  # (7,) float32, clipped
    raw: np.ndarray  # (7,) before clipping
    clipped: bool


def to_env_action(policy_action: torch.Tensor | np.ndarray) -> EnvAction:
    a = policy_action.detach().cpu().numpy() if isinstance(policy_action, torch.Tensor) else np.asarray(policy_action)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape ({ACTION_DIM},) (or (1, {ACTION_DIM})), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"Non-finite action from policy: {a}")
    clipped = np.clip(a, ACTION_LOW, ACTION_HIGH)
    return EnvAction(action=clipped, raw=a, clipped=bool(np.any(clipped != a)))


def summarize_actions(raw_actions: np.ndarray) -> dict[str, float]:
    """Diagnostics for degenerate policies (constant, saturated, or never-closing gripper)."""
    if len(raw_actions) == 0:
        return {}
    acts = np.asarray(raw_actions)
    return {
        "mean_abs_pos": float(np.abs(acts[:, :3]).mean()),
        "mean_abs_rot": float(np.abs(acts[:, 3:6]).mean()),
        "std_pos": float(acts[:, :3].std(axis=0).mean()),
        # Arm saturation matters; gripper "overshoot" (e.g. 1.02 after un-normalization) is benign.
        "clip_frac_arm": float(np.mean(np.any((acts[:, :6] < ACTION_LOW) | (acts[:, :6] > ACTION_HIGH), axis=1))),
        "clip_frac_gripper": float(np.mean((acts[:, 6] < ACTION_LOW) | (acts[:, 6] > ACTION_HIGH))),
        "gripper_close_frac": float(np.mean(acts[:, 6] > 0)),
        "gripper_switches": int(np.sum(np.diff(np.sign(acts[:, 6])) != 0)),
    }
