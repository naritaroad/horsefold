"""Train/test splits by unit or by group, and how often a unit-level split leaks a group into both sides."""

from math import comb

import numpy as np

from horsefold.structure import Structure


def _check_n_test(s: Structure, n_test: int) -> None:
    if not 0 < n_test < s.n_units:
        raise ValueError(f"n_test must be between 1 and {s.n_units - 1}, got {n_test}")


def split_units(rng: np.random.Generator, s: Structure, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Random split of individual units; units of one group can land on both sides."""
    _check_n_test(s, n_test)
    test = np.zeros(s.n_units, bool)
    test[rng.choice(s.n_units, n_test, replace=False)] = True
    return ~test, test


def split_groups(rng: np.random.Generator, s: Structure, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Random split of whole groups; the test set is the prefix of shuffled groups closest to ``n_test`` units."""
    _check_n_test(s, n_test)
    if s.n_groups < 2:
        raise ValueError("a group split needs at least two groups")
    perm = rng.permutation(s.n_groups)
    cum = np.cumsum(s.counts[perm])
    k = int(np.argmin(np.abs(cum[:-1] - n_test))) + 1  # keep at least one group for training
    test = np.isin(s.group, perm[:k])
    return ~test, test


SPLITTERS = {"unit": split_units, "group": split_groups}


def exact_leak_rate(s: Structure, n_test: int) -> dict[str, float]:
    """Share of test units that have another unit of their group in the training set, under ``split_units``.

    A unit from a group of m units leaks unless the other m - 1 all fall in the test set too:
    P(leak) = 1 - C(n_test - 1, m - 1) / C(n_units - 1, m - 1).
    """

    def p_leak(m: int) -> float:
        return 1 - comb(n_test - 1, m - 1) / comb(s.n_units - 1, m - 1)

    m = s.counts[s.group]
    return {
        "all": float(np.mean([p_leak(k) for k in m])),
        "positive": float(np.mean([p_leak(k) for k in m[s.y == 1]])),
        "negative": float(np.mean([p_leak(k) for k in m[s.y == 0]])),
    }


def simulated_leak_rate(rng: np.random.Generator, s: Structure, n_test: int, n_splits: int = 20000) -> dict[str, float]:
    """Monte Carlo check of ``exact_leak_rate``, plus the same-cluster rate (no simple closed form)."""
    same_group, same_cluster = [], []
    for _ in range(n_splits):
        train, test = split_units(rng, s, n_test)
        ti = np.flatnonzero(test)
        same_group.append(np.isin(s.group[ti], s.group[train]))
        same_cluster.append(np.isin(s.cluster[ti], s.cluster[train]))
    return {
        "same_group": float(np.concatenate(same_group).mean()),
        "same_cluster": float(np.concatenate(same_cluster).mean()),
    }
