"""Feature selection, scaling and the classifiers, configured after Basran et al. (2022) Table 2."""

import warnings

import numpy as np
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_NAMES = ("LR", "SVM-Q", "KNN-10", "BT", "RUSBoost", "NN", "1-NN")


def make_models(seed: int, names=MODEL_NAMES) -> dict:
    """Fresh, unfitted classifiers. 1-NN is not in the paper; it shows the extreme case of memorising neighbours."""
    models = {
        "LR": LogisticRegression(penalty=None, max_iter=5000),
        "SVM-Q": SVC(kernel="poly", degree=2, coef0=1, C=1, gamma="scale"),
        "KNN-10": KNeighborsClassifier(10),
        "BT": BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, random_state=seed),
        "RUSBoost": RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_leaf_nodes=21),
            n_estimators=30,
            learning_rate=0.1,
            random_state=seed,
        ),
        "NN": MLPClassifier((25,), activation="relu", alpha=0, max_iter=1000, random_state=seed),
        "1-NN": KNeighborsClassifier(1),
    }
    unknown = set(names) - models.keys()
    if unknown:
        raise ValueError(f"unknown model names: {sorted(unknown)}")
    return {k: v for k, v in models.items() if k in names}


def top_features(X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Indices of the ``n`` features with the largest absolute Welch t statistic between classes.

    The paper ranks features by out-of-bag importance of 100 trees; a t filter is used here for speed.
    """
    a, b = X[y == 1], X[y == 0]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("feature selection needs at least two training rows of each class")
    t = (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    return np.argsort(-np.abs(t))[:n]


def preprocess(X: np.ndarray, y: np.ndarray, ref, n_select: int) -> np.ndarray:
    """Select ``n_select`` features and z-score them, fitting both steps on rows ``ref`` only."""
    feats = top_features(X[ref], y[ref], n_select)
    mu, sd = X[ref][:, feats].mean(0), X[ref][:, feats].std(0) + 1e-9
    return (X[:, feats] - mu) / sd


def scores(model, X: np.ndarray) -> np.ndarray:
    """Continuous scores for AUC: decision function for SVC (no probabilities), probability otherwise."""
    if hasattr(model, "decision_function") and not hasattr(model, "predict_proba"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def fit_quietly(model, X: np.ndarray, y: np.ndarray):
    # Small, noisy training sets trigger convergence and degenerate-fit warnings by design.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(X, y)


def metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    """Accuracy, balanced accuracy and AUC; the last two are NaN when the test set holds one class only."""
    acc = float(np.mean(pred == y))
    if len(np.unique(y)) < 2:
        return acc, np.nan, np.nan
    return acc, float(balanced_accuracy_score(y, pred)), float(roc_auc_score(y, score))


def evaluate_holdout(X, y, train, test, *, leaky: bool, models: dict, n_select: int) -> dict:
    """Fit every model on ``train`` and score it on ``test``.

    ``leaky=True`` fits feature selection and scaling on all rows, test rows included, as happens when
    preprocessing runs before the split.
    """
    Z = preprocess(X, y, slice(None) if leaky else train, n_select)
    out = {}
    for name, m in models.items():
        fit_quietly(m, Z[train], y[train])
        out[name] = metrics(y[test], m.predict(Z[test]), scores(m, Z[test]))
    return out
