# Contributing

Bug reports and questions go to the issue tracker. Please include the command you ran, your Python version and the output of `uv pip list` (or `pip list`).

## Development setup

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The test suite takes about two minutes because it runs small simulations. CI runs the same commands on Ubuntu and Windows with Python 3.12 and 3.14, then a `--quick` end-to-end run.

## Changes

- Keep `results/basran2022.json` and the figures in `docs/assets/` in step with the code: if a change alters the numbers, rerun `uv run horsefold run` (commit the JSON and its `.log`) and `uv run horsefold plot` (and `--lang zh-TW`), and say why the numbers moved in the pull request.
- Add a line under `## [Unreleased]` in `CHANGELOG.md` for every user-visible change.
- A new layout or experiment should come with a test that would fail if it were built wrong, like the counts check in `tests/test_structure.py`.
