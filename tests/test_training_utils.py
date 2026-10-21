import pytest
import torch

from vla_sim_poc.training import RetryOnDecodeError


class Flaky(torch.utils.data.Dataset):
    def __init__(self, fail_times: dict[int, int]):
        self.fail_times = dict(fail_times)

    def __len__(self):
        return 10

    def __getitem__(self, i):
        if self.fail_times.get(i, 0) > 0:
            self.fail_times[i] -= 1
            raise RuntimeError("Could not push packet to decoder")
        return i


def test_transient_error_is_retried_on_the_same_sample():
    assert RetryOnDecodeError(Flaky({3: 2}), retries=3)[3] == 3


def test_persistent_error_substitutes_another_sample():
    out = RetryOnDecodeError(Flaky({3: 100}), retries=2, seed=0)[3]
    assert out != 3 and 0 <= out < 10


def test_non_runtime_errors_propagate():
    class Broken(Flaky):
        def __getitem__(self, i):
            raise KeyError(i)

    with pytest.raises(KeyError):
        RetryOnDecodeError(Broken({}))[0]
