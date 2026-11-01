import numpy as np
import pytest

from horsefold import basran2022, exact_leak_rate, simulated_leak_rate, split_groups, split_units


def test_unit_split_has_exact_test_size():
    s = basran2022()
    train, test = split_units(np.random.default_rng(0), s, 39)
    assert test.sum() == 39
    assert (train ^ test).all()


def test_group_split_never_shares_a_group():
    s = basran2022()
    rng = np.random.default_rng(0)
    for _ in range(200):
        train, test = split_groups(rng, s, 39)
        assert not set(s.group[train]) & set(s.group[test])
        assert abs(int(test.sum()) - 39) <= 2  # groups hold 1-4 units


def test_exact_leak_rate_matches_simulation():
    s = basran2022()
    exact = exact_leak_rate(s, 39)
    sim = simulated_leak_rate(np.random.default_rng(3), s, 39, n_splits=4000)
    assert sim["same_group"] == pytest.approx(exact["all"], abs=0.01)
    assert sim["same_cluster"] < sim["same_group"]


def test_exact_leak_rate_known_values():
    # Two bones per case horse: 1 - 38/128; controls mostly four bones.
    exact = exact_leak_rate(basran2022(), 39)
    assert exact["positive"] == pytest.approx(1 - 38 / 128)
    assert exact["all"] == pytest.approx(0.826, abs=0.001)


def test_singleton_groups_never_leak():
    from horsefold import Structure

    s = Structure.from_groups([(i % 2, [1]) for i in range(20)])
    assert exact_leak_rate(s, 5)["all"] == 0
