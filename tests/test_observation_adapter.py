import numpy as np
import pytest
import torch
from robosuite.utils.transform_utils import quat2axisangle
from scipy.spatial.transform import Rotation

from vla_sim_poc.observation_adapter import (
    STATE_KEY,
    make_env_preprocessor,
    raw_to_policy_frame,
    validate_frame,
)

IMG = "observation.images.image"
WRIST = "observation.images.image2"


def fake_raw_obs(quat_xyzw, size=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "pixels": {
            "image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
            "image2": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        },
        "robot_state": {
            "eef": {"pos": np.array([0.1, -0.2, 0.3]), "quat": np.asarray(quat_xyzw, float), "mat": np.eye(3)},
            "gripper": {"qpos": np.array([0.04, -0.04]), "qvel": np.zeros(2)},
            "joints": {"pos": np.zeros(7), "vel": np.zeros(7)},
        },
    }


def random_quats(n, seed=0):
    return Rotation.random(n, random_state=seed).as_quat()  # (x, y, z, w)


def test_images_rotated_180_scaled_and_channel_first():
    raw = fake_raw_obs([0, 0, 0, 1])
    frame = raw_to_policy_frame(raw, "do the task", make_env_preprocessor())
    for key, cam in ((IMG, "image"), (WRIST, "image2")):
        out = frame[key]
        assert out.shape == (1, 3, 8, 8) and out.dtype == torch.float32
        expected = raw["pixels"][cam][::-1, ::-1].astype(np.float32) / 255.0  # 180 deg rotation
        np.testing.assert_allclose(out[0].permute(1, 2, 0).numpy(), expected, atol=1e-6)
    assert frame["task"] == ["do the task"]


def test_state_layout_pos_axisangle_gripper():
    q = random_quats(1, seed=3)[0]
    frame = raw_to_policy_frame(fake_raw_obs(q), "t", make_env_preprocessor())
    state = frame[STATE_KEY][0].numpy()
    assert state.shape == (8,)
    np.testing.assert_allclose(state[:3], [0.1, -0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(state[6:], [0.04, -0.04], atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_axis_angle_matches_robosuite_convention(seed):
    """The dataset states were produced with robosuite's quat2axisangle; we must match it exactly,
    including its non-canonical handling of w < 0 (angle in (pi, 2pi])."""
    pre = make_env_preprocessor()
    for q in random_quats(20, seed=seed):
        for sign in (1, -1):  # q and -q are the same rotation but give different axis-angle here
            state = raw_to_policy_frame(fake_raw_obs(sign * q), "t", pre)[STATE_KEY][0].numpy()
            np.testing.assert_allclose(state[3:6], quat2axisangle(sign * q), atol=1e-4)


def test_axis_angle_is_a_true_rotation_vector_when_w_positive():
    pre = make_env_preprocessor()
    for q in random_quats(20, seed=11):
        q = q if q[3] >= 0 else -q
        state = raw_to_policy_frame(fake_raw_obs(q), "t", pre)[STATE_KEY][0].numpy()
        np.testing.assert_allclose(state[3:6], Rotation.from_quat(q).as_rotvec(), atol=1e-4)


def test_validate_frame_catches_mismatches():
    pre = make_env_preprocessor()
    frame = raw_to_policy_frame(fake_raw_obs([0, 0, 0, 1], size=8), "t", pre)
    validate_frame(frame, [IMG, WRIST], image_size=8)
    with pytest.raises(ValueError):
        validate_frame(frame, [IMG, WRIST], image_size=256)  # wrong resolution
    with pytest.raises(KeyError):
        validate_frame(frame, ["observation.images.camera1"], image_size=8)  # un-renamed key
    frame[STATE_KEY] = frame[STATE_KEY][:, :6]
    with pytest.raises(ValueError):
        validate_frame(frame, [IMG, WRIST], image_size=8)  # wrong state dim
