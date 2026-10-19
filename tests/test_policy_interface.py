"""Integration test: the real checkpoint's processors and forward pass on a synthetic frame.

Skipped without CUDA. Downloads the pinned checkpoint on first run (~1 GB).
"""

import copy

import numpy as np
import pytest
import torch

from vla_sim_poc.action_adapter import to_env_action
from vla_sim_poc.config import load_config
from vla_sim_poc.observation_adapter import make_env_preprocessor, raw_to_policy_frame
from vla_sim_poc.policy import load_policy

from .test_observation_adapter import fake_raw_obs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def bundle():
    cfg = load_config("eval.yaml", "baseline.yaml")
    return load_policy(cfg["checkpoint"], cfg["policy"])


@pytest.fixture
def frame():
    return raw_to_policy_frame(fake_raw_obs([0, 0, 0, 1], size=256), "pick up the bowl", make_env_preprocessor())


def test_preprocessor_renames_cameras_and_normalizes_state(bundle, frame):
    batch = bundle.preprocessor(copy.deepcopy(frame))
    assert set(bundle.image_keys) == {"observation.images.camera1", "observation.images.camera2"}
    for k in bundle.image_keys:
        assert k in batch and batch[k].shape == (1, 3, 256, 256)
    assert "observation.images.image" not in batch
    # MEAN_STD normalization: state z=0.3 -> (0.3 - 0.7645) / 0.3785
    assert batch["observation.state"][0, 2].item() == pytest.approx((0.3 - 0.76455) / 0.37852, abs=1e-3)


def test_action_chunking_and_output_schema(bundle, frame):
    bundle.reset()
    acts = [to_env_action(bundle.act(copy.deepcopy(frame))) for _ in range(bundle.config.n_action_steps + 1)]
    # one inference fills the queue with n_action_steps actions; the next call triggers a new chunk
    assert len(bundle.inference_latencies_s) == 2
    raw = np.stack([a.raw for a in acts])
    assert raw.shape == (bundle.config.n_action_steps + 1, 7)
    assert np.isfinite(raw).all()
    assert raw[:, :6].std() > 1e-4, "actions are constant: degenerate policy output"


def test_missing_rename_fails_loudly(bundle, frame):
    """Without the camera rename (lerobot#4578), SmolVLA must refuse rather than act blind."""
    from lerobot.processor import RenameObservationsProcessorStep

    step = next(s for s in bundle.preprocessor.steps if isinstance(s, RenameObservationsProcessorStep))
    saved = step.rename_map
    try:
        step.rename_map = {}
        bundle.reset()
        with pytest.raises(ValueError, match="image features are missing"):
            bundle.act(copy.deepcopy(frame))
    finally:
        step.rename_map = saved
