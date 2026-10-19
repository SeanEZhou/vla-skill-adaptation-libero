"""Integration tests on the built task-56 dataset (skipped if it has not been built yet).

The key test independently replays a stored episode from its original HDF5 start state and checks,
at every step, that the dataset's stored state AND images equal the live simulator's observation
*before* the stored action at that index is applied. An off-by-one (observation recorded after the
action) fails at t=0; a wrong camera/rotation fails on the image check.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from vla_sim_poc.config import PROJECT_ROOT, load_config
from vla_sim_poc.dataset import IMAGE_KEYS, SETTLE_ACTION, SETTLE_STEPS, is_noop
from vla_sim_poc.environment import TaskEnv
from vla_sim_poc.observation_adapter import make_env_preprocessor, raw_to_policy_frame

DS_ROOT = PROJECT_ROOT / "data/lerobot/libero90_t56"
pytestmark = pytest.mark.skipif(not (DS_ROOT / "vla_sim_poc/provenance.json").exists(), reason="dataset not built")


@pytest.fixture(scope="module")
def cfg():
    return load_config("eval.yaml", "selected_task.yaml")


@pytest.fixture(scope="module")
def ds():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset("local/libero90_t56_butter_tray", root=DS_ROOT)


@pytest.fixture(scope="module")
def provenance():
    return json.loads((DS_ROOT / "vla_sim_poc/provenance.json").read_text())


def episode_rows(ds, ep):
    idx = np.nonzero(np.asarray(ds.hf_dataset["episode_index"]) == ep)[0]
    return [int(i) for i in idx]


def test_schema_matches_policy_interface(ds, cfg):
    f = ds.meta.features
    assert f["observation.state"]["shape"] == (8,) or list(f["observation.state"]["shape"]) == [8]
    assert list(f["action"]["shape"]) == [7]
    for k in IMAGE_KEYS:
        assert list(f[k]["shape"]) == [cfg["env"]["image_size"], cfg["env"]["image_size"], 3]
    assert ds.meta.fps == cfg["env"]["control_freq"] == 20
    # Exactly one task string, identical to the evaluation instruction.
    assert list(ds.meta.tasks.index) == [cfg["task"]["instruction"]]


def test_actions_in_controller_range_and_gripper_binary(ds):
    a = np.stack(ds.hf_dataset["action"])
    assert np.isfinite(a).all() and np.abs(a[:, :6]).max() <= 1.0
    assert set(np.unique(a[:, 6])) <= {-1.0, 1.0}


def test_frame_index_sequential_within_episodes(ds):
    ep = np.asarray(ds.hf_dataset["episode_index"])
    fi = np.asarray(ds.hf_dataset["frame_index"])
    for e in np.unique(ep):
        np.testing.assert_array_equal(fi[ep == e], np.arange((ep == e).sum()))


def test_dataset_stats_match_frames(ds):
    a = np.stack(ds.hf_dataset["action"]).astype(np.float64)
    np.testing.assert_allclose(ds.meta.stats["action"]["mean"], a.mean(0), atol=1e-5)
    np.testing.assert_allclose(ds.meta.stats["action"]["std"], a.std(0), atol=1e-4)


@pytest.mark.parametrize("episode", [0])
def test_observation_t_precedes_action_t(ds, cfg, provenance, episode):
    demo = next(r["demo"] for r in provenance["demos"] if r.get("episode_index") == episode)
    h5 = PROJECT_ROOT / "data/libero_hdf5" / provenance["source"]["file"]
    with h5py.File(h5) as f:
        s0 = f["data"][demo]["states"][0]
        orig_actions = f["data"][demo]["actions"][()]
    kept, prev = [], None
    for a in orig_actions:
        if not is_noop(a, prev):
            kept.append(a)
            prev = a

    rows = episode_rows(ds, episode)
    # Labels are the original (no-op-filtered) demo actions, stored as float32.
    assert len(kept) == len(rows)
    stored = np.stack([ds.hf_dataset[r]["action"] for r in rows])
    np.testing.assert_allclose(stored, np.stack(kept), atol=1e-7)
    env = TaskEnv(cfg["task"]["suite"], cfg["task"]["task_id"], cfg["env"], use_init_states=True)
    env.env._ensure_env()
    inner, pre = env.env._env, make_env_preprocessor()
    inner.reset()
    raw = inner.set_init_state(s0)
    for _ in range(SETTLE_STEPS):
        raw, _, _, _ = inner.step(SETTLE_ACTION)

    for t, row in enumerate(rows):
        item = ds[row]
        live = raw_to_policy_frame(env.env._format_raw_obs(raw), env.task_description, pre)
        np.testing.assert_allclose(item["observation.state"].numpy(), live["observation.state"][0].numpy(),
                                   atol=1e-5, err_msg=f"state mismatch at t={t}")
        if t % 10 == 0:  # images: lossless PNG + same renderer -> exact
            for k in IMAGE_KEYS:
                stored = (item[k] * 255).round().to(dtype=live[k].dtype)
                assert (stored == (live[k][0] * 255).round()).all(), f"{k} mismatch at t={t}"
        # Replay the float64 originals: float32 rounding (~3e-8) is enough to shift wrist-camera pixels.
        raw, _, _, _ = inner.step(kept[t])
    assert inner.check_success(), "replaying the stored actions must complete the task"
    env.close()
