"""Monte Carlo replicates and the experiment suite for the Basran et al. (2022) case study."""

import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, KFold

from horsefold.data import simulate_features
from horsefold.pipeline import (
    MODEL_NAMES,
    evaluate_holdout,
    fit_quietly,
    make_models,
    metrics,
    preprocess,
    scores,
)
from horsefold.splits import SPLITTERS, exact_leak_rate, simulated_leak_rate
from horsefold.structure import Structure, basran2022

SEED = 20261102


@dataclass(frozen=True)
class Design:
    """Simulation settings. Defaults follow Basran et al. (2022): 102 features, 39 test bones, top 20 kept."""

    d: float = 0.0
    r: float = 0.7
    group_share: float = 5 / 7
    position_sd: float = 0.0
    p: int = 102
    k_signal: int = 5
    n_select: int = 20
    n_test: int = 39

    def features(self, rng, s):
        return simulate_features(
            rng,
            s,
            p=self.p,
            k_signal=self.k_signal,
            d=self.d,
            r=self.r,
            group_share=self.group_share,
            position_sd=self.position_sd,
        )


def _seed(rng) -> int:
    return int(rng.integers(2**31))


# ---------- one replicate each; return {(split, prep, model): (acc, balanced_acc, auc)} ----------
def holdout_rep(seed, s: Structure, design: Design, splits, preps, models) -> dict:
    """One feature draw, one train/test split per split type, every preprocessing mode."""
    rng = np.random.default_rng(seed)
    X, y = design.features(rng, s), s.y
    res = {}
    for sp in splits:
        train, test = SPLITTERS[sp](rng, s, design.n_test)
        for prep in preps:
            fitted = make_models(_seed(rng), models)
            out = evaluate_holdout(X, y, train, test, leaky=prep == "leaky", models=fitted, n_select=design.n_select)
            for name, v in out.items():
                res[(sp, prep, name)] = v
    return res


def cv_rep(seed, s: Structure, design: Design, splits, preps, models) -> dict:
    """10-fold CV by unit (KFold) or by group (GroupKFold); preprocessing inside folds; pooled-prediction metrics."""
    rng = np.random.default_rng(seed)
    X, y = design.features(rng, s), s.y
    res = {}
    for sp in splits:
        if sp == "unit":
            folds = KFold(10, shuffle=True, random_state=_seed(rng)).split(X)
        else:
            folds = GroupKFold(10, shuffle=True, random_state=_seed(rng)).split(X, y, groups=s.group)
        pred = {k: np.zeros(len(y)) for k in models}
        score = {k: np.zeros(len(y)) for k in models}
        for tr_i, te_i in folds:
            train = np.zeros(len(y), bool)
            train[tr_i] = True
            Z = preprocess(X, y, train, design.n_select)
            for name, m in make_models(_seed(rng), models).items():
                fit_quietly(m, Z[tr_i], y[tr_i])
                pred[name][te_i] = m.predict(Z[te_i])
                score[name][te_i] = scores(m, Z[te_i])
        for name in models:
            res[(sp, "clean", name)] = metrics(y, pred[name], score[name])
    return res


def best_of_rep(seed, s: Structure, design: Design, splits, preps, models, n_values=(3, 5, 10, 20)) -> dict:
    """Group 10-fold CV over models x n_select; compare the mean of all configurations with the best one."""
    rng = np.random.default_rng(seed)
    X, y = design.features(rng, s), s.y
    folds = list(GroupKFold(10, shuffle=True, random_state=_seed(rng)).split(X, y, groups=s.group))
    acc = []
    for n in n_values:
        pred = {k: np.zeros(len(y)) for k in models}
        for tr_i, te_i in folds:
            Z = preprocess(X, y, tr_i, n)
            for name, m in make_models(_seed(rng), models).items():
                fit_quietly(m, Z[tr_i], y[tr_i])
                pred[name][te_i] = m.predict(Z[te_i])
        acc += [float(np.mean(pred[name] == y)) for name in models]
    k = len(acc)
    return {
        ("group", "clean", f"mean_of_{k}"): (float(np.mean(acc)), np.nan, np.nan),
        ("group", "clean", f"best_of_{k}"): (float(np.max(acc)), np.nan, np.nan),
    }


# ---------- run many replicates in parallel and summarise ----------
def run(
    tag: str,
    reps: int,
    s: Structure,
    design: Design,
    *,
    fn: Callable = holdout_rep,
    splits: Sequence[str] = ("unit", "group"),
    preps: Sequence[str] = ("clean",),
    models: Sequence[str] = ("LR", "SVM-Q", "KNN-10", "1-NN"),
    base_seed: int = SEED,
    n_jobs: int = -1,
) -> dict:
    """Run ``reps`` replicates of ``fn`` and summarise each (split, prep, model) cell.

    Replicate seeds come from ``SeedSequence([base_seed, crc32(tag)])``, so a scenario gives the same numbers
    regardless of ``n_jobs`` or which other scenarios run.
    """
    if reps < 2:
        raise ValueError("reps must be at least 2")
    if unknown := set(splits) - SPLITTERS.keys():
        raise ValueError(f"unknown splits: {sorted(unknown)}; use 'unit' and/or 'group'")
    if unknown := set(preps) - {"clean", "leaky"}:
        raise ValueError(f"unknown preps: {sorted(unknown)}; use 'clean' and/or 'leaky'")
    if fn is not holdout_rep and tuple(preps) != ("clean",):
        raise ValueError("cv_rep and best_of_rep fit preprocessing inside folds; use preps=('clean',)")
    if fn is best_of_rep and tuple(splits) != ("group",):
        raise ValueError("best_of_rep always uses group folds; use splits=('group',)")
    seeds = np.random.SeedSequence([base_seed, zlib.crc32(tag.encode())]).generate_state(reps)
    out = Parallel(n_jobs=n_jobs)(delayed(fn)(int(x), s, design, splits, preps, models) for x in seeds)
    return {"|".join(key): summarise(np.array([o[key] for o in out])) for key in out[0]}


def summarise(a: np.ndarray) -> dict:
    """``a`` is reps x (acc, balanced_acc, auc). MCSE = SD / sqrt(reps); percentiles show single-split spread."""
    acc = a[:, 0]

    def mean_or_none(col):
        return None if np.isnan(col).all() else round(float(np.nanmean(col)), 3)

    return {
        "acc": round(float(acc.mean()), 3),
        "acc_mcse": round(float(acc.std(ddof=1) / np.sqrt(len(acc))), 4),
        "acc_p2.5": round(float(np.percentile(acc, 2.5)), 3),
        "acc_p97.5": round(float(np.percentile(acc, 97.5)), 3),
        "bal_acc": mean_or_none(a[:, 1]),
        "auc": mean_or_none(a[:, 2]),
    }


def basran2022_suite(scale: float = 1.0, n_jobs: int = -1, progress: Callable[[str], None] = print) -> dict:
    """Experiments E1-E6 on the Basran et al. (2022) layout. ``scale`` multiplies every replicate count."""
    import importlib.metadata as md

    import horsefold

    s = basran2022()
    base = Design()
    replicates = {}

    def go(tag, reps, structure=s, **kw):
        t0 = time.time()
        design = replace(base, **{k: kw.pop(k) for k in list(kw) if k in Design.__dataclass_fields__})
        n = replicates[tag] = max(2, round(reps * scale))
        res = run(tag, n, structure, design, n_jobs=n_jobs, **kw)
        progress(f"{tag}: {n} replicates, {time.time() - t0:.0f} s")
        return res

    leak = {"exact_same_group": {k: round(v, 3) for k, v in exact_leak_rate(s, base.n_test).items()}}
    leak.update({k: round(v, 3) for k, v in simulated_leak_rate(np.random.default_rng(SEED), s, base.n_test).items()})
    progress(f"leak rate: {leak}")

    res = {
        "meta": {
            "horsefold": horsefold.__version__,
            "seed": SEED,
            "scale": scale,
            "replicates": replicates,  # filled in as each scenario runs
            "design": asdict(base),
            "structure": s.summary(),
            "versions": {p: md.version(p) for p in ("numpy", "scikit-learn", "imbalanced-learn", "joblib")},
            "source": "Basran et al. (2022) Animals 12(21):3033, doi:10.3390/ani12213033; p.3 Fig. 1, Tables 2 and 4",
        },
        "leak_rate": leak,
    }
    # E1: 2 splits x 2 preprocessing modes x 7 models; d = 0 (no signal) and d = 1.0.
    res["E1_main"] = {
        f"d={d}": go(f"E1 d={d}", 500, d=d, preps=("clean", "leaky"), models=MODEL_NAMES) for d in (0.0, 1.0)
    }
    # E2: within-cluster correlation sweep with no signal; r = 0 is the negative control.
    res["E2_r_sweep"] = {f"r={r}": go(f"E2 r={r}", 300, r=r) for r in (0.0, 0.3, 0.5, 0.7, 0.9)}
    # E3: signal strength sweep at r = 0.7.
    res["E3_d_sweep"] = {
        f"d={d}": go(f"E3 d={d}", 300, d=d, models=("LR", "SVM-Q")) for d in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    }
    # E4: sensitivity to the assumptions the paper does not pin down.
    res["E4_sensitivity"] = {
        "missing_unit_on_single_cluster_group": go("E4 missing", 300, structure=basran2022("single")),
        "cluster_dominant_group_share_0.3": go("E4 limb", 300, group_share=0.3),
        "position_offset_sd_0.5": go("E4 type", 300, position_sd=0.5),
    }
    # E5: 10-fold CV, which the paper also reports.
    res["E5_10fold_cv"] = {
        f"d={d}": go(f"E5 d={d}", 200, d=d, models=("LR", "SVM-Q", "KNN-10"), fn=cv_rep) for d in (0.0, 1.0)
    }
    # E6: optimism of reporting the best of 24 configurations (6 models x 4 feature counts).
    res["E6_best_of_configs"] = {
        f"d={d}": go(f"E6 d={d}", 200, d=d, splits=("group",), models=MODEL_NAMES[:6], fn=best_of_rep)
        for d in (0.0, 1.0)
    }
    return res
