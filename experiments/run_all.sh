#!/usr/bin/env bash
# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
#
# Reproduce the experiments of the paper, stage by stage.
#
# Usage (from the experiments/ folder):
#   ./run_all.sh <stage> [<stage> ...]
#
# Stages, in pipeline order:
#   data           Download the 12 OpenML datasets                       (minutes, network-dependent)
#   train          Train linear/logistic, RF, MLP and XGBoost models      (minutes)
#   analytic       Precompute closed-form optima for linear/logistic      (< 1 h)
#   tuning         Gold standards + NSGA-II hyperparameter tuning         (~12 h)
#   convergence    Analytic convergence study (Sec. 7.2)                  (~3 h)
#   audit          Comparative audit against baselines (Sec. 7.3)         (~1-2 h)
#   computational  Complexity and dimensionality study (Sec. 7.4)         (< 30 min)
#   example        Running example of Section 3                           (< 1 min)
#   perturbation   Perturbation realization heatmaps                      (< 1 min)
#   figures        Regenerate every paper figure and table from results/  (minutes)
#   all            Run every stage above in order
#
# Approximate runtimes on an Apple M5 Pro (48 GB RAM) with WORKERS=5; with the default
# of 1 worker, the parallel stages (analytic, tuning, convergence, audit) take several
# times longer. The "figures" stage only needs the result files included in the
# repository (see README.md), plus the trained models for the perturbation heatmaps.
#
# Parallel stages use WORKERS processes (default 1; each loads its own copy of the
# models and datasets, so raise it only if RAM allows), e.g. WORKERS=8 ./run_all.sh audit

set -euo pipefail
cd "$(dirname "$0")"

# With uv, scripts run in the locked uv environment; without uv, they run with the
# active Python environment (e.g. a venv set up with pip, see README.md).
if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
else
  PYTHON=(python)
fi

run() {
  echo ">>> ${PYTHON[*]} $*"
  "${PYTHON[@]}" "$@"
}

stage_data() { run shared/datasets.py; }
stage_train() { run shared/training.py --bucket all; }
# --force-datasets all recomputes the optima instead of resuming from the included file.
stage_analytic() { run shared/precompute_analytic.py --workers "${WORKERS:-1}" --force-datasets all; }

stage_tuning() {
  run hyperparameter_tuning/generate_gold_standards.py --workers "${WORKERS:-1}"
  run hyperparameter_tuning/hyperparameter_tuning.py --bucket all --workers "${WORKERS:-1}"
}

stage_convergence() { run analytic_convergence/analytic_convergence.py --workers "${WORKERS:-1}"; }
stage_audit() { run comparative_audit/comparative_audit.py --skip-visualize --workers "${WORKERS:-1}"; }

stage_computational() {
  run computational_study/computational_study.py
  run computational_study/dimensionality_scaling.py --model rf --force
  run computational_study/benchmark_unit_inference.py
}

stage_example() { run running_example/running_example.py; }

stage_perturbation() {
  for alpha in 0.10 0.90; do
    run perturbation_heatmap/run_experiment.py \
      --dataset cpu_act --model-family random_forest --instance-idx 4928 \
      --alpha "$alpha" --n-realizations 40 \
      --mu 0.25 --sigma 0.10 --env-std 0.02 --mode normalized
  done
}

stage_figures() {
  run running_example/visualize_and_report.py
  run analytic_convergence/visualize_and_report.py
  run comparative_audit/visualize_and_report.py
  run computational_study/visualize_and_report.py
  run hyperparameter_tuning/visualize_and_report.py
  run collect_paper_figures.py
}

stage_all() {
  stage_data
  stage_train
  stage_analytic
  stage_tuning
  stage_convergence
  stage_audit
  stage_computational
  stage_example
  stage_perturbation
  stage_figures
}

if [[ $# -eq 0 ]]; then
  sed -n '5,29p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

for stage in "$@"; do
  if ! declare -F "stage_${stage}" >/dev/null; then
    echo "Unknown stage: ${stage}" >&2
    exit 1
  fi
  "stage_${stage}"
done
