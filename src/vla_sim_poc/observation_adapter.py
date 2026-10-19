"""Raw LIBERO observation -> SmolVLA policy input.

The conversion is LeRobot's own code path (same one ``lerobot-eval`` uses); this module only
sequences it and validates the result against what the checkpoint declares:

1. ``preprocess_observation``: HWC uint8 images -> BCHW float32 in [0, 1], keys
   ``observation.images.image`` (agentview) / ``observation.images.image2`` (wrist); nested
   ``robot_state`` passed through.
2. ``LiberoProcessorStep``: rotate images 180 deg (LIBERO renders upside down; the training data was
   stored upright) and build the 8-D state ``[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]``.
3. Attach the language instruction as ``task``.

The policy's own preprocessor (applied later in ``policy.py``) then renames cameras to
``camera1``/``camera2``, tokenizes the instruction and MEAN_STD-normalizes the state.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep

STATE_KEY = "observation.state"
STATE_LAYOUT = ("eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az", "gripper_q0", "gripper_q1")


def make_env_preprocessor() -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline(steps=[LiberoProcessorStep()])


def add_batch_dim(raw_obs: dict[str, Any]) -> dict[str, Any]:
    """LiberoEnv returns unbatched arrays; LeRobot's pipeline expects a leading batch dim."""
    import numpy as np

    def rec(x):
        if isinstance(x, dict):
            return {k: rec(v) for k, v in x.items()}
        if isinstance(x, np.ndarray):
            return x[None]
        return x

    return rec(raw_obs)


def raw_to_policy_frame(
    raw_obs: dict[str, Any], task_description: str, env_preprocessor: PolicyProcessorPipeline
) -> dict[str, Any]:
    frame = preprocess_observation(add_batch_dim(raw_obs))
    frame = env_preprocessor(frame)
    frame["task"] = [task_description]
    return frame


def validate_frame(frame: dict[str, Any], expected_image_keys: list[str], image_size: int) -> None:
    """Fail loudly on the silent-mismatch class of bugs before the policy ever sees the frame."""
    for key in expected_image_keys:
        if key not in frame:
            raise KeyError(f"Missing image key {key!r}; frame has {sorted(k for k in frame if 'image' in k)}")
        img = frame[key]
        if tuple(img.shape) != (1, 3, image_size, image_size):
            raise ValueError(f"{key}: expected (1, 3, {image_size}, {image_size}), got {tuple(img.shape)}")
        if img.dtype != torch.float32 or img.min() < 0.0 or img.max() > 1.0:
            raise ValueError(f"{key}: expected float32 in [0,1], got {img.dtype} [{img.min()}, {img.max()}]")
    state = frame[STATE_KEY]
    if tuple(state.shape) != (1, len(STATE_LAYOUT)):
        raise ValueError(f"state: expected (1, {len(STATE_LAYOUT)}), got {tuple(state.shape)}")
    if not torch.isfinite(state).all():
        raise ValueError(f"state contains non-finite values: {state}")
