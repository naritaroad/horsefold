import numpy as np
import pytest

from horsefold import Structure, simulate_features


def _pair_corr(X, i, j):
    return np.corrcoef(X[i], X[j])[0, 1]


def test_correlation_structure_matches_parameters():
    # One group with two clusters of two units; many features act as many draws.
    s = Structure.from_groups([(0, [2, 2])])
    X = simulate_features(np.random.default_rng(1), s, p=200_000, k_signal=0, r=0.7, group_share=5 / 7)
    assert _pair_corr(X, 0, 1) == pytest.approx(0.7, abs=0.01)  # same cluster
    assert _pair_corr(X, 0, 2) == pytest.approx(0.5, abs=0.01)  # same group, other cluster: 0.7 * 5/7
    assert X.var() == pytest.approx(1.0, abs=0.01)


def test_signal_only_on_first_k_features():
    s = Structure.from_groups([(1, [1])] * 2000 + [(0, [1])] * 2000)
    X = simulate_features(np.random.default_rng(2), s, p=10, k_signal=3, d=1.0, r=0.0)
    diff = X[s.y == 1].mean(0) - X[s.y == 0].mean(0)
    assert np.allclose(diff[:3], 1.0, atol=0.1)
    assert np.allclose(diff[3:], 0.0, atol=0.1)


@pytest.mark.parametrize("kw", [{"r": 1.2}, {"r": -0.1}, {"group_share": 2}, {"k_signal": 200}])
def test_rejects_out_of_range(kw):
    s = Structure.from_groups([(0, [1])])
    with pytest.raises(ValueError):
        simulate_features(np.random.default_rng(0), s, **kw)
