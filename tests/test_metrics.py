import pytest

from vla_sim_poc.metrics import mcnemar_exact, paired_comparison, proportion, wilson_interval


def test_wilson_known_value():
    # Reference: 8/10 -> Wilson 95% CI (0.4902, 0.9433)
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_wilson_edges_are_informative():
    lo, hi = wilson_interval(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0.05 < hi < 0.09  # 0/50 is not "certainly 0%"
    lo, hi = wilson_interval(50, 50)
    assert hi == 1.0 and 0.91 < lo < 0.95


def test_proportion():
    p = proportion([True, False, True, True])
    assert (p.successes, p.n, p.rate) == (3, 4, 0.75)


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # 0 vs 6 discordant: p = 2 * 0.5**6
    assert mcnemar_exact(0, 6) == pytest.approx(2 * 0.5**6)


def test_paired_comparison_counts_discordant_pairs():
    a = [True, False, False, True, False]
    b = [True, True, True, False, False]
    c = paired_comparison(a, b, n_boot=500)
    assert (c.only_a, c.only_b, c.n_pairs) == (1, 2, 5)
    assert c.diff == pytest.approx(0.2)
    assert c.bootstrap_ci[0] <= c.diff <= c.bootstrap_ci[1]


def test_paired_comparison_rejects_misaligned():
    with pytest.raises(ValueError):
        paired_comparison([True], [True, False])
