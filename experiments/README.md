# Reproducing the Experiments

This folder contains every script needed to reproduce the numerical results, figures, and tables of the paper *Directional Counterfactual Explanations* and its electronic companion.

- [1. Requirements](#1-requirements)
- [2. Installation](#2-installation)
- [3. Quick check (minutes)](#3-quick-check-minutes)
- [4. Regenerating figures and tables from precomputed results](#4-regenerating-figures-and-tables-from-precomputed-results)
- [5. Full pipeline from scratch](#5-full-pipeline-from-scratch)
- [6. Map of paper figures and tables](#6-map-of-paper-figures-and-tables)
- [7. Random seeds](#7-random-seeds)
- [8. Folder layout](#8-folder-layout)

---

## 1. Requirements

| Component | Version used |
|---|---|
| Hardware | Apple M5 Pro, 48 GB RAM (CPU only; no GPU required) |
| Operating system | macOS (any Linux or macOS with the tools below should work) |
| Python | 3.13 (pinned in `experiments/pyproject.toml`; the `dicex` package itself supports ≥ 3.12) |
| [uv](https://docs.astral.sh/uv/) | 0.11.8 |
| Rust toolchain | 1.96.0 (pinned in `rust-toolchain.toml`; needed to build the `dicex` Rust extension) |

Exact versions of every Python dependency are locked in `uv.lock` (package) and `experiments/uv.lock` (experiments). Internet access is required once, to download the datasets from OpenML.

## 2. Installation

The experiments have their own locked environment (`experiments/pyproject.toml`, `experiments/uv.lock`), with Python 3.13 and the `dicex` package of this repository. Choose one of the following options; all commands are run from the repository root.

**Option A: uv with the prebuilt wheel from PyPI (no Rust needed).**

```bash
cd experiments
uv sync --no-install-package dicex
uv pip install --no-deps dicex==0.1.0
export UV_NO_SYNC=1   # PowerShell: $env:UV_NO_SYNC = "1"
```

`UV_NO_SYNC=1` keeps `uv run` (and `run_all.sh`) from re-syncing the environment, which would rebuild `dicex` from source; set it in every new shell. Version 0.1.0 is the code used for the paper.

**Option B: uv, building `dicex` from source (requires a [Rust toolchain](https://rustup.rs); the pinned version is selected automatically).**

```bash
cd experiments
uv sync
```

`uv sync` creates `experiments/.venv`, installs the locked dependencies, and builds the local `dicex` package (Python + Rust extension).

**Option C: pip.** Requires Python 3.13. `experiments/requirements.txt` pins the same versions as `experiments/uv.lock`:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r experiments/requirements.txt
pip install dicex==0.1.0
```

Option C is meant for machines without uv: `run_all.sh` runs the scripts with the active environment only when uv is not installed (with uv installed, it uses `uv run`, which ignores this environment; use Option A instead).

All commands below are run from the `experiments/` folder.

## 3. Quick check (minutes)

The running example of Section 3 uses synthetic data and needs no download or training:

```bash
./run_all.sh example figures
```

This regenerates Figures 1–3 (under a minute for the example; the `figures` stage then redraws every figure for which results are available). Collected figures are written to `experiments/paper_figures/` under the same file names used in the manuscript. The two perturbation heatmaps, which need the trained models, are reported as missing with a warning (see Section 4).

## 4. Regenerating figures and tables from precomputed results

This repository includes the raw result files of every long-running experiment:

| File | Produced by | Used for |
|---|---|---|
| `analytic_convergence/results/raw_results_<timestamp>.csv` (most recent run) | `analytic_convergence/analytic_convergence.py` | Fig. convergence errors; Tables presets and convergence summary |
| `comparative_audit/results/paper_results.csv` | `comparative_audit/comparative_audit.py` | Fig. heatmaps, Fig. risk–reward; Tables audit summary, model split, noise, percentile; Table base times |
| `hyperparameter_tuning/results/optuna/*_trials_*.jsonl` | `hyperparameter_tuning/hyperparameter_tuning.py` | EC Pareto figures |
| `hyperparameter_tuning/results/selected_configs/*.yaml` | `hyperparameter_tuning/hyperparameter_tuning.py` | EC Table selected configurations; optimizer settings of the audit and computational study |
| `computational_study/results/{model_complexity.json, dimensionality_rf.json, unit_inference_benchmark.csv}` | `computational_study/*.py` | Tables complexity overhead and dimensionality scaling |
| `models/metrics.csv` | `shared/training.py` | EC Tables model performance |
| `models/analytic_solutions.pkl` | `shared/precompute_analytic.py` | Closed-form optima of the linear and logistic models, read by stages 4 and 5 (so stage 3 can be skipped) |

With these files in place:

```bash
./run_all.sh example figures
```

regenerates every figure and table of the paper in a few minutes, except the two perturbation-realization heatmaps, which need the trained models (downloading the data and training the models takes a few minutes):

```bash
./run_all.sh data train perturbation figures
```

## 5. Full pipeline from scratch

`run_all.sh` runs each stage in order; stages can also be run individually. The runtimes below are orders of magnitude measured on the hardware above with 5 workers (`WORKERS=5`); with the default of 1 worker, the parallel stages (3–6) take several times longer. They are given only to plan a re-run.

| # | Stage | Command(s) | Output | Approx. runtime (5 workers) |
|---|---|---|---|---|
| 1 | `data` | `shared/datasets.py` | `data/raw`, `data/interim` (12 OpenML datasets) | minutes (network-dependent) |
| 2 | `train` | `shared/training.py --bucket all` | `models/*.joblib`, `models/metrics.csv` | minutes |
| 3 | `analytic` | `shared/precompute_analytic.py --force-datasets all` | `models/analytic_solutions.pkl` (linear/logistic optima) | < 1 h (peak ~18 GB) |
| 4 | `tuning` | `hyperparameter_tuning/generate_gold_standards.py`, `hyperparameter_tuning/hyperparameter_tuning.py --bucket all` | `hyperparameter_tuning/results/{gold_standards, optuna, selected_configs}` | ~12 h |
| 5 | `convergence` | `analytic_convergence/analytic_convergence.py` | `analytic_convergence/results/raw_results_<timestamp>.csv` | ~3 h (peak ~20 GB) |
| 6 | `audit` | `comparative_audit/comparative_audit.py` | `comparative_audit/results/{raw_results_<timestamp>.csv, paper_results.csv}` | ~1–2 h (peak ~31 GB) |
| 7 | `computational` | `computational_study/computational_study.py`, `dimensionality_scaling.py --model rf --force`, `benchmark_unit_inference.py` | `computational_study/results/*.json`, `*.csv` | < 30 min (single process) |
| 8 | `example` | `running_example/running_example.py` | `running_example/results/running_example_data.pkl` | < 1 min |
| 9 | `perturbation` | `perturbation_heatmap/run_experiment.py` (two runs, α = 0.10 and 0.90) | `perturbation_heatmap/results/plots/` | < 1 min |
| 10 | `figures` | all `visualize_and_report*.py` scripts, then `collect_paper_figures.py` | `*/results/plots`, `*/results/tables`, `paper_figures/` | minutes |

Dependencies between stages: 2 needs 1; 3 needs 2; 4 and 5 need 2 and the closed-form optima of 3, which are included in `models/analytic_solutions.pkl` (so 4 and 5 can run without re-running 3; they stop with an error if the file is missing); 6 needs 2; 6 and 7 read the configurations selected in 4, which are included; 9 needs 2.

Notes:

- **Memory.** The long-running stages are parallelized with Python's `multiprocessing` to iterate faster on the machine above, which favors speed over memory efficiency. Each worker process (and the main process) loads its own copy of the trained models and datasets, so memory grows with the number of workers. All parallel scripts and `run_all.sh` default to 1 worker, which needs about 11–15 GB of RAM for the comparative audit (the most demanding stage); set `WORKERS=N` (or `--workers N`) to run faster if memory allows. Each additional worker adds roughly 5 GB (audit), 3.5 GB (convergence), or 3 GB (analytic precomputation). The results in the paper were produced with 5 workers on a 48 GB machine; they do not depend on the number of workers.
- Stage 5 and the visualization of stage 10 use the most recent `raw_results_*.csv` in `analytic_convergence/results/`. If any task of stage 5 fails, its partial results are saved as `incomplete_raw_results_<timestamp>.csv`, which the reporting scripts ignore, and the stage exits with an error.
- `computational_study/dimensionality_scaling.py` resumes by default: it skips the configurations already saved in `computational_study/results/dimensionality_rf.json` (included). Stage 7 of `run_all.sh` passes `--force`, which recomputes them all. Likewise, `shared/precompute_analytic.py` resumes from `models/analytic_solutions.pkl`, and stage 3 passes `--force-datasets all` to recompute it.
- Stage 6 always refreshes `paper_results.csv`, which the reporting scripts read. To report a specific run, pass `--input-file` to `comparative_audit/visualize_and_report.py`.
- The hyperparameter tuning (stage 4, reported in the electronic companion) was run once to select the optimizer configurations shipped in `hyperparameter_tuning/results/selected_configs/`; all other experiments use these configurations as fixed inputs, so stage 4 does not need to be re-run to reproduce Sections 7.2–7.4.
- The optimizer presets (`minimal`, `low`, `mid`, `high`, `xhigh`) and the eight noise scenarios of the convergence study are defined in `analytic_convergence/experiment_config.yaml`; the comparative audit uses four of them.

### Output logs of the reported runs

The logs of the runs that produced the included results are in the `results/logs/` folder of each experiment (absolute paths of the original machine have been removed). Re-running a stage writes a new timestamped log next to them.

| Stage | Log file(s) |
|---|---|
| `analytic` | `hyperparameter_tuning/results/logs/precompute_analytic_20260929_203730.log` |
| `tuning` (gold standards: high, low, medium buckets) | `hyperparameter_tuning/results/logs/generate_gold_standards_20260623_{033436,104831,110840}.log` |
| `tuning` (NSGA-II: high, low, medium buckets) | `hyperparameter_tuning/results/logs/hyperparameter_tuning_20260623_{043459,114353,130942}.log` |
| `convergence` | `analytic_convergence/results/logs/analytic_convergence_20260929_211452.log` |
| `audit` | `comparative_audit/results/logs/comparative_audit_20260930_001405.log` |
| `computational` | `computational_study/results/logs/{computational_study_20261001_142505,dimensionality_scaling_20261001_142633,benchmark_unit_inference_20261001_144648}.log` |

## 6. Map of paper figures and tables

### Figures

All files are collected into `paper_figures/` by `collect_paper_figures.py`.

| Paper figure | File(s) | Generated by |
|---|---|---|
| Execution uncertainty along the Wachter CF direction (Sec. 3) | `x0_wachter_cf_scaled.pdf`, `x0_wachter_cf_direction_samples_scaled.pdf` | `running_example/visualize_and_report.py` |
| Comparison of candidate directions (Sec. 3) | `directional_comparison.pdf`, `improvement_distribution.pdf` | `running_example/visualize_and_report.py` |
| Optimal direction fields (Sec. 3) | `vector_field_wachter_cf.pdf`, `vector_field_dicex_alpha_0_9.pdf`, `vector_field_dicex_alpha_0_1.pdf` | `running_example/visualize_and_report.py` |
| Perturbation realizations (Sec. 6) | `heatmap_cpu_act_random_forest_idx4928_alpha{10,90}_n40_normalized.pdf` | `perturbation_heatmap/run_experiment.py` |
| Convergence errors (Sec. 7.2) | `convergence_angular_error.pdf`, `convergence_relative_error.pdf` | `analytic_convergence/visualize_and_report.py` |
| Heatmaps across noise and α (Sec. 7.3) | `heatmap_cvar_gain.pdf`, `heatmap_fail_rate_reduction.pdf`, `heatmap_volatility_reduction.pdf` | `comparative_audit/visualize_and_report.py` |
| Risk–reward trade-off (Sec. 7.3) | `pareto_frontier.pdf` | `comparative_audit/visualize_and_report.py` |
| EC: Pareto fronts of the hyperparameter tuning | `pareto_frontier_{low,medium,high}.pdf` | `hyperparameter_tuning/visualize_and_report.py` |

### Tables

| Paper table (label) | Source |
|---|---|
| Noise scenarios (`tab:noise-scenarios`) | `analytic_convergence/experiment_config.yaml` |
| Optimizer presets (`tab:presets`) | `analytic_convergence/results/plots/paper/tables/tab_presets.csv` |
| Convergence summary (`tab:convergence-summary`) | `analytic_convergence/results/plots/paper/tables/tab_convergence_summary.csv` |
| Value regression reported in Sec. 7.2 (slopes, intercepts, $R^2$) | `analytic_convergence/results/plots/paper/tables/tab_value_regression.csv` |
| Overall comparison (`tab:audit-grand-summary`) | `comparative_audit/results/tables/paper/audit_grand_summary.csv` |
| Comparison by model complexity (`tab:audit-model-type-split`) | `comparative_audit/results/tables/paper/audit_model_type_split.csv` |
| Comparison by noise level (`tab:noise-breakdown`) | `comparative_audit/results/tables/paper/audit_noise_breakdown.csv` |
| Comparison by query percentile (`tab:audit-percentile`) | `comparative_audit/results/tables/paper/audit_percentile.csv` |
| Execution times (`tab:base-times`) | `computational_study/results/tables/paper/base_times.csv` |
| Model complexity overhead (`tab:complexity-overhead`) | `computational_study/results/tables/paper/complexity_overhead.csv` |
| Dimensionality scaling (`tab:dim-scaling`) | `computational_study/results/tables/paper/dimensionality_scaling.csv` |
| EC: model performance (`tab:model-performance-classification`, `tab:model-performance-regression`) | `models/metrics.csv` (dataset sizes in `data/DATASETS.md`) |
| EC: selected configurations (`tab:selected-configs`) | `hyperparameter_tuning/results/selected_configs/*.yaml` and `hyperparameter_tuning/results/reports/pareto_front_report.md` |

The tables of the paper report the values of these files rounded as printed.

## 7. Random seeds

| Component | Seed | Location |
|---|---|---|
| Dataset preprocessing and train/test split | 42 | `shared/datasets.py`, `shared/training.py` |
| Model training | 42 | `shared/training.py` |
| Analytic precomputation | 42 | `shared/precompute_analytic.py` |
| Hyperparameter tuning and gold standards | 20260530 | `hyperparameter_tuning/hyperparameter_tuning.py`, `generate_gold_standards.py` |
| Analytic convergence | 20260608 | `analytic_convergence/experiment_config.yaml` |
| Comparative audit | 42 | `comparative_audit/comparative_audit.py` |
| Computational study (data subsample and optimizer) | 42 (dimensionality: 42, 123, 456, 789, 101112, one per synthetic dataset) | `computational_study/*.py` |
| Running example | 42 | `running_example/running_example.py` |
| Perturbation heatmaps | 42 (`--seed`) | `perturbation_heatmap/run_experiment.py` |

The runtimes of the computational study depend on the machine. Its model-evaluation counts are reproducible with the seeds above, up to one caveat: to time realistic multithreaded inference, its Random Forests predict with `n_jobs=-1`, and scikit-learn then sums the tree predictions in a nondeterministic order. The resulting differences (of order 1e-15) can occasionally flip an acceptance decision of the optimizer and change the count of a configuration. In our checks, 9 of the 10 configurations of the complexity study reproduced their counts exactly. The models of all other experiments predict with `n_jobs=1` and are deterministic.

## 8. Folder layout

```
experiments/
├── README.md                  # this file
├── run_all.sh                 # stage-by-stage pipeline
├── collect_paper_figures.py   # gathers the manuscript figures into paper_figures/
├── pyproject.toml, uv.lock    # locked environment of the experiments
├── data/                      # datasets.yaml registry and DATASETS.md
├── shared/                    # datasets, training, analytic solutions, common helpers
├── analytic_convergence/      # Sec. 7.2: convergence against closed-form optima
├── comparative_audit/         # Sec. 7.3: comparison with LIME-derived, nominal and random baselines
├── computational_study/       # Sec. 7.4: execution times, model complexity, dimensionality
├── hyperparameter_tuning/     # EC: dimension-aware NSGA-II tuning of vMF-VNS
├── perturbation_heatmap/      # Sec. 6: realized perturbations and CVaR tail
└── running_example/           # Sec. 3: synthetic credit-score example
```

Datasets are described in [`data/DATASETS.md`](data/DATASETS.md).
