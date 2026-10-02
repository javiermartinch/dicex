# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Collect the figures used in the paper and its electronic companion.

Each experiment writes its plots under its own ``results/`` folder. This script copies
the exact files used in the manuscript into ``experiments/paper_figures/`` under the file
names referenced by the LaTeX sources, and reports any figure that has not been generated.

Usage:
    uv run python collect_paper_figures.py [--output-dir DIR] [--strict]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys

EXPERIMENTS_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger("collect_paper_figures")

# Paper file name -> (source path relative to experiments/, generating script).
PAPER_FIGURES: dict[str, tuple[str, str]] = {
    # Section 3: running example.
    "x0_wachter_cf_scaled.pdf": (
        "running_example/results/plots/x0_wachter_cf_scaled.pdf",
        "running_example/visualize_and_report.py",
    ),
    "x0_wachter_cf_direction_samples_scaled.pdf": (
        "running_example/results/plots/x0_wachter_cf_direction_samples_scaled.pdf",
        "running_example/visualize_and_report.py",
    ),
    "directional_comparison.pdf": (
        "running_example/results/plots/directional_comparison.pdf",
        "running_example/visualize_and_report.py",
    ),
    "improvement_distribution.pdf": (
        "running_example/results/plots/improvement_distribution.pdf",
        "running_example/visualize_and_report.py",
    ),
    "vector_field_wachter_cf.pdf": (
        "running_example/results/plots/vector_field_wachter_cf.pdf",
        "running_example/visualize_and_report.py",
    ),
    "vector_field_dicex_alpha_0_9.pdf": (
        "running_example/results/plots/vector_field_dicex_alpha_0_9.pdf",
        "running_example/visualize_and_report.py",
    ),
    "vector_field_dicex_alpha_0_1.pdf": (
        "running_example/results/plots/vector_field_dicex_alpha_0_1.pdf",
        "running_example/visualize_and_report.py",
    ),
    # Section 6: realized perturbations (cpu_act, random forest).
    "heatmap_cpu_act_random_forest_idx4928_alpha10_n40_normalized.pdf": (
        "perturbation_heatmap/results/plots/heatmap_cpu_act_random_forest_idx4928_alpha10_n40_normalized.pdf",
        "perturbation_heatmap/run_experiment.py",
    ),
    "heatmap_cpu_act_random_forest_idx4928_alpha90_n40_normalized.pdf": (
        "perturbation_heatmap/results/plots/heatmap_cpu_act_random_forest_idx4928_alpha90_n40_normalized.pdf",
        "perturbation_heatmap/run_experiment.py",
    ),
    # Section 7.2: convergence on exact benchmarks.
    "convergence_angular_error.pdf": (
        "analytic_convergence/results/plots/paper/boxplots/convergence_angular_error.pdf",
        "analytic_convergence/visualize_and_report.py",
    ),
    "convergence_relative_error.pdf": (
        "analytic_convergence/results/plots/paper/boxplots/convergence_relative_error.pdf",
        "analytic_convergence/visualize_and_report.py",
    ),
    # Section 7.3: comparative audit.
    "heatmap_cvar_gain.pdf": (
        "comparative_audit/results/plots/paper/heatmap_cvar_gain.pdf",
        "comparative_audit/visualize_and_report.py",
    ),
    "heatmap_fail_rate_reduction.pdf": (
        "comparative_audit/results/plots/paper/heatmap_fail_rate_reduction.pdf",
        "comparative_audit/visualize_and_report.py",
    ),
    "heatmap_volatility_reduction.pdf": (
        "comparative_audit/results/plots/paper/heatmap_volatility_reduction.pdf",
        "comparative_audit/visualize_and_report.py",
    ),
    "pareto_frontier.pdf": (
        "comparative_audit/results/plots/paper/pareto_frontier.pdf",
        "comparative_audit/visualize_and_report.py",
    ),
    # Electronic companion: hyperparameter tuning.
    "pareto_frontier_low.pdf": (
        "hyperparameter_tuning/results/optuna/pareto_plots/pareto_frontier_low.pdf",
        "hyperparameter_tuning/visualize_and_report.py",
    ),
    "pareto_frontier_medium.pdf": (
        "hyperparameter_tuning/results/optuna/pareto_plots/pareto_frontier_medium.pdf",
        "hyperparameter_tuning/visualize_and_report.py",
    ),
    "pareto_frontier_high.pdf": (
        "hyperparameter_tuning/results/optuna/pareto_plots/pareto_frontier_high.pdf",
        "hyperparameter_tuning/visualize_and_report.py",
    ),
}


def main() -> None:
    """Copy every paper figure into the output directory under its manuscript name (CLI entry point)."""
    parser = argparse.ArgumentParser(description="Collect the figures used in the paper.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS_ROOT / "paper_figures",
        help="Destination directory (default: experiments/paper_figures).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any paper figure is missing.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for paper_name, (source, script) in PAPER_FIGURES.items():
        source_path = EXPERIMENTS_ROOT / source
        if not source_path.exists():
            missing.append((paper_name, script))
            continue
        shutil.copy2(source_path, args.output_dir / paper_name)
        logger.info("[ok]      %s", paper_name)

    for paper_name, script in missing:
        logger.warning("[missing] %s  (run %s)", paper_name, script)

    logger.info("%d/%d figures copied to %s", len(PAPER_FIGURES) - len(missing), len(PAPER_FIGURES), args.output_dir)
    if missing:
        # Missing figures are expected when only some stages were run (e.g. the perturbation
        # heatmaps need the trained models), so they are reported without failing.
        logger.warning("%d figure(s) not generated yet; run the scripts listed above to add them.", len(missing))
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
