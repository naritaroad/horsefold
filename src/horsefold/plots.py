"""Figures from a results JSON written by ``horsefold run``. Needs matplotlib (``pip install horsefold[plot]``)."""

import json
from pathlib import Path

PAPER, INK, MUTED, GRID = "#F7F5F0", "#15171C", "#7D838D", "#E3DFD6"
RED, GREEN, BLUE, GOLD = "#C8412B", "#2F6B4F", "#2B5C8A", "#B8862F"
# Families that ship regular and bold weights first; some Noto Sans TC installs carry only a thin variable face.
CJK_FONTS = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC", "Noto Sans TC"]

TEXT = {
    "en": {
        "models": {
            "LR": "Logistic\nregression",
            "SVM-Q": "Quadratic\nSVM",
            "KNN-10": "KNN-10",
            "BT": "Bagged\ntrees",
            "RUSBoost": "RUSBoost",
            "NN": "Neural\nnet",
            "1-NN": "1-NN\n(extreme)",
        },
        "short": {"LR": "Logistic regression", "SVM-Q": "Quadratic SVM", "KNN-10": "KNN-10"},
        "unit_leaky": "Random {unit} split + features chosen on all data",
        "unit_clean": "Random {unit} split (features chosen on training set)",
        "group_clean": "Split by {group} (correct)",
        "unit": "Random {unit} split",
        "group": "Split by {group}",
        "chance": "chance 0.5",
        "f1_y": "Test accuracy (mean of {n} runs)",
        "f1_title": "With no real signal, reported accuracy depends on how the data were split",
        "f2_x": "Correlation r between {unit}s of the same {cluster} ({group}-level share {share})",
        "f2_y": "Test accuracy (mean of {n} runs)",
        "f2_title": "No signal: the more alike a {group}'s {unit}s, the higher a {unit}-level split scores",
        "f2_solid": "{unit}-level split (solid)",
        "f2_dashed": "split by {group} (dashed)",
        "f3_x": "Signal strength d (shift on {k} features, in SD)",
        "f3_y": "Test accuracy (logistic regression)",
        "f3_band": " (line: mean; band: 95% of single splits)",
        "f3_title": "Same signal, higher reported accuracy with a {unit}-level split",
    },
    "zh-TW": {
        "models": {
            "LR": "邏輯斯迴歸",
            "SVM-Q": "二次 SVM",
            "KNN-10": "KNN-10",
            "BT": "Bagged\nTrees",
            "RUSBoost": "RUSBoost",
            "NN": "神經網路",
            "1-NN": "1-NN\n（極端示範）",
        },
        "short": {"LR": "邏輯斯迴歸", "SVM-Q": "二次 SVM", "KNN-10": "KNN-10"},
        "unit_leaky": "隨機切分{unit}＋用全部資料挑特徵",
        "unit_clean": "隨機切分{unit}（只用訓練集挑特徵）",
        "group_clean": "以{group}分組（正確做法）",
        "unit": "隨機切分{unit}",
        "group": "以{group}分組",
        "chance": "瞎猜 0.5",
        "f1_y": "測試集準確率（{n} 次模擬平均）",
        "f1_title": "特徵與標籤完全無關時，不同切分方式報出的準確率",
        "f2_x": "同一{cluster}兩塊{unit}的特徵相關 r（{group}層級占 {share}）",
        "f2_y": "測試集準確率（{n} 次平均）",
        "f2_title": "沒有真訊號時，同{group}的{unit}越像，隨機切分的準確率越高",
        "f2_solid": "隨機切分{unit}（實線）",
        "f2_dashed": "以{group}分組（虛線）",
        "f3_x": "真訊號強度 d（{k} 個特徵上的位移，以標準差計）",
        "f3_y": "測試集準確率（邏輯斯迴歸）",
        "f3_band": "（線：平均；色帶：單次切分 95% 範圍）",
        "f3_title": "同樣的真訊號，隨機切分{unit}報出的準確率較高",
    },
}
NAMES_ZH = {"horse": "馬", "limb": "肢", "bone": "骨頭", "group": "組", "cluster": "簇", "unit": "樣本"}


def _setup(lang: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    rc = {
        "font.size": 12,
        "axes.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titleweight": "bold",
        "axes.titlesize": 13.5,
        "legend.frameon": False,
    }
    if lang == "zh-TW":
        installed = {f.name for f in font_manager.fontManager.ttflist}
        found = [f for f in CJK_FONTS if f in installed]
        if not found:
            raise RuntimeError(f"zh-TW figures need one of these fonts installed: {', '.join(CJK_FONTS)}")
        rc["font.family"] = found[:1]
    return plt, rc


def make_figures(results: dict | str | Path, out_dir: str | Path, lang: str = "en") -> list[Path]:
    """Write F1 (main comparison), F2 (correlation sweep) and F3 (signal sweep) as PNG; return their paths.

    ``results`` must come from ``basran2022_suite`` (``horsefold run``); figures read its E1-E3 scenarios.
    Styling is scoped to this call and does not change global matplotlib settings.
    """
    if lang not in TEXT:
        raise ValueError(f"lang must be one of {list(TEXT)}")
    if not isinstance(results, dict):
        results = json.loads(Path(results).read_text(encoding="utf-8"))
    plt, rc = _setup(lang)
    with plt.rc_context(rc):
        return _draw(results, out_dir, lang, plt)


def _draw(results: dict, out_dir, lang: str, plt) -> list[Path]:
    import numpy as np

    t = TEXT[lang]
    g, c, u = results["meta"]["structure"]["names"]
    if lang == "zh-TW":
        g, c, u = (NAMES_ZH.get(x, x) for x in (g, c, u))
    names = {"group": g, "cluster": c, "unit": u}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if lang == "en" else f".{lang}"
    paths = []

    def save(fig, stem):
        p = out_dir / f"{stem}{suffix}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        paths.append(p)

    # F1: three bars per model at d = 0
    e1 = results["E1_main"]["d=0.0"]
    models = [m for m in t["models"] if f"unit|clean|{m}" in e1]
    series = [
        ("unit|leaky", RED, t["unit_leaky"]),
        ("unit|clean", GOLD, t["unit_clean"]),
        ("group|clean", GREEN, t["group_clean"]),
    ]
    gap = [0.45 if m == "1-NN" else 0 for m in models]
    xs = np.arange(len(models)) + np.array(gap)
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for j, (key, col, lab) in enumerate(series):
        vals = [e1[f"{key}|{m}"]["acc"] for m in models]
        ax.bar(xs + (j - 1) * w, vals, w - 0.03, color=col, label=lab.format(**names))
        for x, v in zip(xs + (j - 1) * w, vals, strict=True):
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(xs[-1] + 0.5, 0.505, t["chance"], va="bottom", ha="left", fontsize=9, color=MUTED)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 1.2)
    ax.set_xticks(xs, [t["models"][m] for m in models], fontsize=10)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel(t["f1_y"].format(n=results["meta"]["replicates"]["E1 d=0.0"]))
    ax.legend(loc="upper left", fontsize=9.5, ncol=2)
    ax.set_title(t["f1_title"], loc="left")
    save(fig, "F1_main")

    # F2: r sweep, d = 0
    e2 = results["E2_r_sweep"]
    rs = [float(k.split("=")[1]) for k in e2]
    share = round(results["meta"]["design"]["group_share"], 2)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for m, col in (("SVM-Q", RED), ("LR", BLUE), ("KNN-10", GOLD)):
        b = [e2[k][f"unit|clean|{m}"]["acc"] for k in e2]
        h = [e2[k][f"group|clean|{m}"]["acc"] for k in e2]
        ax.plot(rs, b, color=col, lw=2, marker="o", ms=6)
        ax.plot(rs, h, color=col, lw=1.5, ls="--", marker="o", ms=5, mfc=PAPER)
        ax.text(rs[-1] + 0.02, b[-1], t["short"][m], va="center", fontsize=9.5, color=INK)
    ax.plot([], [], color=INK, lw=2, label=t["f2_solid"].format(**names))
    ax.plot([], [], color=INK, lw=1.5, ls="--", label=t["f2_dashed"].format(**names))
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set_xlim(-0.03, rs[-1] + 0.22)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks(rs)
    ax.set_xlabel(t["f2_x"].format(share=share, **names))
    ax.set_ylabel(t["f2_y"].format(n=results["meta"]["replicates"]["E2 r=0.0"]))
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title(t["f2_title"].format(**names), loc="left")
    save(fig, "F2_r_sweep")

    # F3: d sweep for logistic regression with single-split spread
    e3 = results["E3_d_sweep"]
    ds = [float(k.split("=")[1]) for k in e3]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for key, col, lab in (("unit|clean|LR", GOLD, t["unit"]), ("group|clean|LR", GREEN, t["group"])):
        mean = [e3[k][key]["acc"] for k in e3]
        lo = [e3[k][key]["acc_p2.5"] for k in e3]
        hi = [e3[k][key]["acc_p97.5"] for k in e3]
        ax.fill_between(ds, lo, hi, color=col, alpha=0.13, lw=0)
        ax.plot(ds, mean, color=col, lw=2, marker="o", ms=6, label=lab.format(**names) + t["f3_band"])
    ax.set_xlim(-0.05, ds[-1] + 0.05)
    ax.set_ylim(0.25, 1.02)
    ax.set_xticks(ds)
    ax.set_xlabel(t["f3_x"].format(k=results["meta"]["design"]["k_signal"]))
    ax.set_ylabel(t["f3_y"])
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(t["f3_title"].format(**names), loc="left")
    save(fig, "F3_d_sweep")
    return paths
