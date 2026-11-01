# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog 2.0.0](https://keepachangelog.com/en/2.0.0/), and the project uses [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Structure` for three-level layouts (group → cluster → unit) and the `basran2022()` layout: 129 bones, 65 limbs, 50 horses.
- Synthetic features with set within-cluster and within-group correlation and an optional true signal.
- Unit-level and group-level splits, with the unit-level leak rate in closed form and by simulation.
- The Basran et al. (2022) pipeline: Welch *t* feature selection, z-scoring, and six classifiers plus 1-NN, with preprocessing fitted on training rows or on all rows.
- Experiments E1–E6 (main comparison, correlation sweep, signal sweep, sensitivity, 10-fold CV, best-of-configurations optimism) with seeds that do not depend on the worker count.
- `horsefold run`, `horsefold plot` and `horsefold leak-rate` commands; `--quick` for a smoke run.
- `horsefold run` saves its progress output, with timestamps, Python and library versions and timings, to a log next to the JSON (`--log` to change or skip).
- Figures F1–F3 in English and Traditional Chinese.
