import numpy as np
import pytest
import torch

from vla_sim_poc.action_adapter import ACTION_DIM, summarize_actions, to_env_action


def test_accepts_batched_tensor_and_returns_float32_vector():
    a = to_env_action(torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.01, -0.01, 1.0]]))
    assert a.action.shape == (ACTION_DIM,) and a.action.dtype == np.float32
    assert not a.clipped


def test_clips_out_of_range_and_reports_it():
    a = to_env_action(np.array([1.5, -2, 0, 0, 0, 0, 1], dtype=np.float32))
    assert a.clipped
    np.testing.assert_array_equal(a.action[:2], [1.0, -1.0])
    np.testing.assert_array_equal(a.raw[:2], [1.5, -2.0])


@pytest.mark.parametrize("bad", [np.zeros(6), np.zeros((2, 7)), np.zeros(32)])
def test_rejects_wrong_shapes(bad):
    # e.g. a 6-D SO-100 action (smolvla_base) or an unsliced 32-D padded action
    with pytest.raises(ValueError):
        to_env_action(bad)


def test_rejects_non_finite():
    with pytest.raises(ValueError):
        to_env_action(np.array([np.nan, 0, 0, 0, 0, 0, 1]))


def test_summary_flags_gripper_behaviour():
    acts = np.zeros((10, 7))
    acts[:, 6] = [-1] * 4 + [1] * 6
    s = summarize_actions(acts)
    assert s["gripper_close_frac"] == pytest.approx(0.6)
    assert s["gripper_switches"] == 1
    assert s["clip_frac_arm"] == 0.0 and s["clip_frac_gripper"] == 0.0
