import json

import pytest

from horsefold.cli import main


@pytest.fixture(scope="module")
def tiny_results(tmp_path_factory):
    out = tmp_path_factory.mktemp("res") / "tiny.json"
    assert main(["run", "--scale", "0.001", "--out", str(out)]) == 0
    return out


def test_run_writes_all_experiments(tiny_results):
    res = json.loads(tiny_results.read_text(encoding="utf-8"))
    assert {
        "meta",
        "leak_rate",
        "E1_main",
        "E2_r_sweep",
        "E3_d_sweep",
        "E4_sensitivity",
        "E5_10fold_cv",
        "E6_best_of_configs",
    } <= res.keys()
    assert res["meta"]["structure"]["units"] == 129
    assert res["leak_rate"]["exact_same_group"]["all"] == 0.826


def test_run_writes_progress_log(tiny_results):
    log = tiny_results.with_suffix(".log").read_text(encoding="utf-8")
    assert "E1 d=0.0: 2 replicates" in log
    assert "E6 d=1.0" in log and "total" in log


def test_plot_writes_three_figures(tiny_results, tmp_path):
    pytest.importorskip("matplotlib")
    assert main(["plot", str(tiny_results), "--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "F1_main.png",
        "F2_r_sweep.png",
        "F3_d_sweep.png",
    ]


def test_leak_rate_command(capsys):
    assert main(["leak-rate", "--splits", "500"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["exact_same_group"]["all"] == pytest.approx(0.826, abs=0.001)
