import numpy as np
import pytest

from horsefold import Design, Structure, basran2022, run, split_groups, split_units
from horsefold.experiments import best_of_rep, cv_rep
from horsefold.pipeline import top_features


@pytest.mark.parametrize("n_test", [0, 129, 200])
def test_splits_reject_impossible_test_sizes(n_test):
    s = basran2022()
    for split in (split_units, split_groups):
        with pytest.raises(ValueError):
            split(np.random.default_rng(0), s, n_test)


def test_group_split_always_keeps_a_training_group():
    s = Structure.from_groups([(1, [3]), (0, [3])])
    for seed in range(20):
        train, test = split_groups(np.random.default_rng(seed), s, 5)
        assert train.any() and test.any()


def test_top_features_needs_two_rows_per_class():
    with pytest.raises(ValueError):
        top_features(np.zeros((3, 4)), np.array([1.0, 0.0, 0.0]), 2)


@pytest.mark.parametrize(
    "kw",
    [
        {"splits": ("bone",)},
        {"preps": ("dirty",)},
        {"fn": cv_rep, "preps": ("leaky",)},
        {"fn": best_of_rep, "splits": ("unit", "group")},
    ],
)
def test_run_rejects_unsupported_options(kw):
    with pytest.raises(ValueError):
        run("x", 2, basran2022(), Design(), **kw)
