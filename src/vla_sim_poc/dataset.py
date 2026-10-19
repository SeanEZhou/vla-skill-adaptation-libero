"""Build a LeRobot dataset for one LIBERO task by replaying the official HDF5 demonstrations
through the *same* observation path the policy sees at evaluation time.

Recipe (mirrors OpenVLA's ``regenerate_libero_dataset.py``, which produced the data
``lerobot/smolvla_libero`` was trained on):

1. ``env.reset()``, ``set_init_state(demo.states[0])``, then 10 no-op settle steps (gripper open).
   This is exactly what ``LiberoEnv.reset`` does at evaluation.
2. For each recorded action: skip it if it is a no-op (arm delta ~0 and gripper unchanged), else
   **record the current observation, then step the action**. Frame t is the observation the action
   at index t was chosen from.
3. Keep the episode only if the replay ends in task success.

Observations go through ``LiberoEnv._format_raw_obs`` -> ``raw_to_policy_frame`` (180 deg rotation,
8-D state), so the stored images/state are byte-for-byte what evaluation feeds the policy.
Images are stored losslessly (PNG), unlike the AV1-encoded ``lerobot/libero``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .environment import TaskEnv
from .observation_adapter import make_env_preprocessor, raw_to_policy_frame

NOOP_THRESHOLD = 1e-4
SETTLE_STEPS = 10
SETTLE_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def lerobot_features(image_size: int) -> dict:
    """Same keys/shapes as ``lerobot/libero`` (what smolvla_libero was trained on), images lossless."""
    img = {"dtype": "image", "shape": (image_size, image_size, 3), "names": ["height", "width", "channel"]}
    return {
        IMAGE_KEYS[0]: dict(img),
        IMAGE_KEYS[1]: dict(img),
        STATE_KEY: {"dtype": "float32", "shape": (8,), "names": ["state"]},
        ACTION_KEY: {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }


def is_noop(action: np.ndarray, prev_action: np.ndarray | None, threshold: float = NOOP_THRESHOLD) -> bool:
    """OpenVLA's no-op rule: arm delta ~0 AND gripper command unchanged from the previous kept action.
    (A still arm that opens/closes the gripper is *not* a no-op.)"""
    if prev_action is None:
        return bool(np.linalg.norm(action[:-1]) < threshold)
    return bool(np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1])


@dataclass
class RegeneratedEpisode:
    demo_key: str
    images: dict[str, list[np.ndarray]] = field(default_factory=lambda: {k: [] for k in IMAGE_KEYS})
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    sim_states: list[np.ndarray] = field(default_factory=list)  # MuJoCo state before each kept action
    n_noops: int = 0
    success: bool = False
    first_success_frame: int | None = None

    def __len__(self) -> int:
        return len(self.actions)


def _to_hwc_uint8(chw) -> np.ndarray:
    return (chw.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def regenerate_episode(env: TaskEnv, demo_key: str, init_sim_state: np.ndarray, actions: np.ndarray) -> RegeneratedEpisode:
    env.env._ensure_env()
    inner = env.env._env
    pre = make_env_preprocessor()
    ep = RegeneratedEpisode(demo_key)

    inner.reset()
    raw = inner.set_init_state(init_sim_state)
    for _ in range(SETTLE_STEPS):
        raw, _, _, _ = inner.step(SETTLE_ACTION)

    prev = None
    for a in np.asarray(actions, dtype=np.float64):
        if is_noop(a, prev):
            ep.n_noops += 1
            continue
        frame = raw_to_policy_frame(env.env._format_raw_obs(raw), env.task_description, pre)
        for k in IMAGE_KEYS:
            ep.images[k].append(_to_hwc_uint8(frame[k][0]))
        ep.states.append(frame[STATE_KEY][0].numpy().astype(np.float32))
        ep.actions.append(a.astype(np.float32))
        ep.sim_states.append(inner.sim.get_state().flatten().copy())
        raw, _, done, _ = inner.step(a)
        prev = a
        if ep.first_success_frame is None and inner.check_success():
            ep.first_success_frame = len(ep.actions)
    ep.success = bool(inner.check_success())
    return ep


def episode_split(n_episodes: int, n_val: int, seed: int) -> tuple[list[int], list[int]]:
    """Episode-level split (never frame-level): a seeded permutation of episode indices."""
    if not 0 <= n_val < n_episodes:
        raise ValueError(f"n_val={n_val} must be in [0, {n_episodes})")
    perm = np.random.default_rng(seed).permutation(n_episodes)
    val = sorted(int(i) for i in perm[:n_val])
    train = sorted(int(i) for i in perm[n_val:])
    return train, val


def compute_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0)}


def stats_shift_in_std(new: dict, ref: dict) -> np.ndarray:
    """How far the new data's mean sits from the reference mean, in reference-std units, per dim.
    Large values mean that normalizing with new stats would feed the pretrained net very different
    numbers for the same physical state/action."""
    return np.abs(np.asarray(new["mean"]) - np.asarray(ref["mean"])) / (np.asarray(ref["std"]) + 1e-8)


def episodes_for_task(repo_id: str, revision: str, task_index: int) -> list[int]:
    """Episode -> task mapping from the (small) data parquet files; episode metadata lacks it."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id, repo_type="dataset", revision=revision, allow_patterns=["data/*"]))
    df = pd.concat(pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in sorted(root.glob("data/*/*.parquet")))
    per_ep = df.groupby("episode_index")["task_index"].agg(["min", "max"])
    assert (per_ep["min"] == per_ep["max"]).all(), "an episode spans several tasks"
    return sorted(int(e) for e in per_ep.index[per_ep["min"] == task_index])
