"""Command line: ``horsefold run``, ``horsefold plot`` and ``horsefold leak-rate``."""

import argparse
import json
import sys
from pathlib import Path

QUICK_SCALE = 0.04


def _run(args) -> int:
    import os
    import platform
    import time
    from datetime import datetime

    from horsefold.experiments import basran2022_suite

    scale = QUICK_SCALE if args.quick else args.scale
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = None if args.log == "none" else Path(args.log) if args.log else out.with_suffix(".log")
    log = log_path.open("w", encoding="utf-8") if log_path else None

    def say(msg: str) -> None:
        # Same lines on screen and in the log; the log adds a timestamp. No host name or absolute paths.
        print(msg, file=sys.stderr, flush=True)
        if log:
            log.write(f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}  {msg}\n")
            log.flush()

    t0 = time.time()
    try:
        say(f"horsefold run --scale {scale} --jobs {args.jobs} --out {args.out}")
        say(f"python {platform.python_version()} on {platform.system()} {platform.machine()}, {os.cpu_count()} threads")
        res = basran2022_suite(scale=scale, n_jobs=args.jobs, progress=say)
        say(f"versions: {res['meta']['versions']}")
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        say(f"wrote {args.out}; total {time.time() - t0:.0f} s")
    finally:
        if log:
            log.close()
    return 0


def _plot(args) -> int:
    from horsefold.plots import make_figures

    for p in make_figures(args.results, args.out, lang=args.lang):
        print(f"wrote {p}", file=sys.stderr)
    return 0


def _leak_rate(args) -> int:
    import numpy as np

    from horsefold.splits import exact_leak_rate, simulated_leak_rate
    from horsefold.structure import basran2022

    s = basran2022()
    res = {
        "exact_same_group": exact_leak_rate(s, args.n_test),
        "simulated": simulated_leak_rate(np.random.default_rng(args.seed), s, args.n_test, args.splits),
    }
    print(json.dumps(res, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="horsefold", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run experiments E1-E6 on the Basran et al. (2022) layout and write JSON")
    r.add_argument("--out", default="results/basran2022.json", help="output JSON path")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help=f"smoke run with {QUICK_SCALE:.0%} of the replicates")
    g.add_argument("--scale", type=float, default=1.0, help="multiply every replicate count (default 1.0)")
    r.add_argument("--jobs", type=int, default=-1, help="parallel workers (default: all cores)")
    r.add_argument("--log", help="progress log path (default: next to --out with .log; 'none' to skip)")
    r.set_defaults(func=_run)

    p = sub.add_parser("plot", help="draw figures F1-F3 from a results JSON (needs matplotlib)")
    p.add_argument("results", nargs="?", default="results/basran2022.json")
    p.add_argument("--out", default="docs/assets", help="output folder")
    p.add_argument("--lang", choices=["en", "zh-TW"], default="en")
    p.set_defaults(func=_plot)

    lr = sub.add_parser("leak-rate", help="share of test bones with a same-horse bone in training, exact and simulated")
    lr.add_argument("--n-test", type=int, default=39)
    lr.add_argument("--splits", type=int, default=20000)
    lr.add_argument("--seed", type=int, default=20261102)
    lr.set_defaults(func=_leak_rate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
