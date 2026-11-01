"""Synthetic features with group- and cluster-level correlation."""

import numpy as np

from horsefold.structure import Structure


def simulate_features(
    rng: np.random.Generator,
    s: Structure,
    *,
    p: int = 102,
    k_signal: int = 5,
    d: float = 0.0,
    r: float = 0.7,
    group_share: float = 5 / 7,
    position_sd: float = 0.0,
) -> np.ndarray:
    """Return an ``(n_units, p)`` feature matrix.

    Each feature is the sum of a group effect, a cluster effect and unit noise, each with unit total variance.
    Two units in the same cluster correlate at ``r``; two units in the same group but different clusters
    correlate at ``r * group_share``. Positive units get ``+d`` (in standard deviations) on the first
    ``k_signal`` features; ``d = 0`` means the features carry no information about the label.
    ``position_sd`` adds a systematic offset per within-cluster position (e.g. medial vs lateral),
    shared by both classes.
    """
    if not 0 <= r <= 1:
        raise ValueError("r must be in [0, 1]")
    if not 0 <= group_share <= 1:
        raise ValueError("group_share must be in [0, 1]")
    if not 0 <= k_signal <= p:
        raise ValueError("k_signal must be in [0, p]")
    var_group, var_cluster = r * group_share, r * (1 - group_share)
    X = (
        rng.normal(0, np.sqrt(var_group), (s.n_groups, p))[s.group]
        + rng.normal(0, np.sqrt(var_cluster), (s.n_clusters, p))[s.cluster]
        + rng.normal(0, np.sqrt(1 - r), (s.n_units, p))
    )
    if position_sd:
        X += rng.normal(0, position_sd, (int(s.position.max()) + 1, p))[s.position]
    X[:, :k_signal] += d * s.y[:, None]
    return X
