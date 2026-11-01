import pytest

from horsefold import Design, basran2022, run
from horsefold.experiments import best_of_rep, cv_rep


def test_same_numbers_regardless_of_worker_count():
    s, design = basran2022(), Design()
    a = run("determinism", 4, s, design, models=("LR",), n_jobs=1)
    b = run("determinism", 4, s, design, models=("LR",), n_jobs=2)
    assert a == b


def test_negative_control_r0_shows_no_split_difference():
    # Without within-horse correlation, splitting by bone or by horse should score the same.
    res = run("neg-control", 40, basran2022(), Design(r=0.0), models=("LR",), n_jobs=-1)
    assert res["unit|clean|LR"]["acc"] == pytest.approx(res["group|clean|LR"]["acc"], abs=0.08)


def test_unit_split_inflates_1nn_under_strong_correlation():
    res = run("inflation", 20, basran2022(), Design(r=0.9), models=("1-NN",), n_jobs=-1)
    assert res["unit|clean|1-NN"]["acc"] > 0.75
    assert res["group|clean|1-NN"]["acc"] < 0.65


def test_cv_and_best_of_reps_produce_expected_keys():
    s, design = basran2022(), Design()
    cv = cv_rep(1, s, design, ("unit", "group"), ("clean",), ("LR",))
    assert set(cv) == {("unit", "clean", "LR"), ("group", "clean", "LR")}
    best = best_of_rep(1, s, design, ("group",), ("clean",), ("LR", "KNN-10"), n_values=(3, 5))
    mean, top = best[("group", "clean", "mean_of_4")][0], best[("group", "clean", "best_of_4")][0]
    assert top >= mean


def test_run_rejects_single_replicate():
    with pytest.raises(ValueError):
        run("x", 1, basran2022(), Design())
