import numpy as np
import pytest

from horsefold.pipeline import MODEL_NAMES, evaluate_holdout, make_models, preprocess, top_features


def test_top_features_finds_the_shifted_columns():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(200)]
    X = rng.normal(size=(400, 50))
    X[:, [7, 21]] += 1.5 * y[:, None]
    assert set(top_features(X, y, 2)) == {7, 21}


def test_clean_preprocessing_ignores_test_rows():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(30), np.zeros(30)]
    X = rng.normal(size=(60, 40))
    train = np.zeros(60, bool)
    train[::2] = True
    Z1 = preprocess(X, y, train, 5)
    X2 = X.copy()
    X2[~train] = rng.normal(10, 5, size=X2[~train].shape)  # change only test rows
    Z2 = preprocess(X2, y, train, 5)
    assert np.allclose(Z1[train], Z2[train])


def test_leaky_preprocessing_uses_test_rows():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(30), np.zeros(30)]
    X = rng.normal(size=(60, 40))
    X2 = X.copy()
    X2[:10] += 10
    assert not np.allclose(preprocess(X, y, slice(None), 5), preprocess(X2, y, slice(None), 5))


def test_make_models_filters_and_validates():
    assert list(make_models(0)) == list(MODEL_NAMES)
    assert list(make_models(0, ("LR", "1-NN"))) == ["LR", "1-NN"]
    with pytest.raises(ValueError):
        make_models(0, ("XGB",))


def test_evaluate_holdout_returns_metrics_in_range():
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(40), np.zeros(40)]
    X = rng.normal(size=(80, 30))
    test = np.zeros(80, bool)
    test[rng.choice(80, 20, replace=False)] = True
    out = evaluate_holdout(X, y, ~test, test, leaky=False, models=make_models(0), n_select=10)
    for acc, bacc, auc in out.values():
        assert 0 <= acc <= 1 and 0 <= bacc <= 1 and 0 <= auc <= 1
