import numpy as np
import pytest

from vla_sim_poc.dataset import compute_stats, episode_split, is_noop, stats_shift_in_std


def test_split_is_episode_level_disjoint_and_complete():
    train, val = episode_split(49, n_val=5, seed=0)
    assert len(val) == 5 and len(train) == 44
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(49))


def test_split_is_deterministic_and_seed_dependent():
    assert episode_split(49, 5, seed=0) == episode_split(49, 5, seed=0)
    assert episode_split(49, 5, seed=0)[1] != episode_split(49, 5, seed=1)[1]


def test_split_rejects_bad_sizes():
    with pytest.raises(ValueError):
        episode_split(5, n_val=5, seed=0)


def test_noop_rule_keeps_gripper_only_actions():
    still_open = np.array([0, 0, 0, 0, 0, 0, -1.0])
    still_close = np.array([0, 0, 0, 0, 0, 0, 1.0])
    move = np.array([0.1, 0, 0, 0, 0, 0, -1.0])
    assert is_noop(still_open, None)                # first action, arm still
    assert not is_noop(move, None)
    assert is_noop(still_open, prev_action=still_open)
    assert not is_noop(still_close, prev_action=still_open)  # gripper changes -> keep


def test_stats_shift_in_reference_std_units():
    ref = {"mean": np.array([0.0, 1.0]), "std": np.array([1.0, 0.5])}
    new = compute_stats(np.array([[2.0, 1.0], [2.0, 1.0]]))
    np.testing.assert_allclose(stats_shift_in_std(new, ref), [2.0, 0.0], atol=1e-6)
