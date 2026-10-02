# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Pareto-front plots of the hyperparameter tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.helpers import repo_relative
from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.persistence import ensure_layout, experiment_layout
from experiments.shared.plots import setup_latex_matplotlib as setup_matplotlib

logger = get_logger("visualize_and_report")

FS_LABEL = 20
FS_LEGEND = 16
FS_TICK = 16

BUCKETS_ORDERED: list[tuple[str, str, str]] = [
    ("low", "Low Dimensionality ($d \\le 10$)", "#2b8cbe"),
    ("medium", "Medium Dimensionality ($11 \\le d \\le 50$)", "#e66101"),
    ("high", "High Dimensionality ($d > 50$)", "#7b3294"),
]

BUCKET_TO_COLOR = {k: c for k, _, c in BUCKETS_ORDERED}


def load_optuna_trials(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load Optuna trial JSONL files for each dimension bucket.

    Args:
        results_dir (Path): Directory containing the Optuna results.

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping bucket name (low/medium/high) to a
            DataFrame with trial-level data.
    """
    dfs: dict[str, pd.DataFrame] = {}
    for bucket, _, _ in BUCKETS_ORDERED:
        jsonl_files = sorted(results_dir.glob(f"{bucket}_trials_*.jsonl"))
        if not jsonl_files:
            logger.warning("No trial files found for bucket '%s' in %s", bucket, results_dir)
            continue

        rows: list[dict[str, Any]] = []
        for fp in jsonl_files:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    trial = json.loads(line)
                    row: dict[str, Any] = {
                        "trial_number": trial["trial_number"],
                        "precision_loss": trial["precision_loss"] * 100,
                        "mean_time_seconds": trial["mean_time_seconds"],
                        "mean_n_model_evals": trial["mean_n_model_evals"],
                        "bucket": bucket,
                    }
                    row.update(trial["config"])
                    rows.append(row)
        dfs[bucket] = pd.DataFrame(rows)
        logger.info(
            "Loaded %d trials for bucket '%s' from %s",
            len(dfs[bucket]),
            bucket,
            jsonl_files[0].name,
        )
    return dfs


def compute_pareto_front(
    df: pd.DataFrame,
    x_col: str = "mean_time_seconds",
    y_col: str = "precision_loss",
) -> np.ndarray:
    """Compute the Pareto frontier indices (minimize both objectives).

    A trial dominates another if it has both lower (or equal) x and
    lower (or equal) y, with at least one strictly lower.

    Args:
        df (pd.DataFrame): DataFrame with trial data.
        x_col (str): Column name for the x-axis objective (cost).
        y_col (str): Column name for the y-axis objective (error).

    Returns:
        np.ndarray: Boolean array indicating which rows are on the Pareto frontier.
    """
    vals = np.asarray(df[[x_col, y_col]].to_numpy(), dtype=float)
    n = len(vals)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (vals[j, 0] <= vals[i, 0] and vals[j, 1] <= vals[i, 1]) and (
                vals[j, 0] < vals[i, 0] or vals[j, 1] < vals[i, 1]
            ):
                is_pareto[i] = False
                break

    return is_pareto


def plot_pareto_frontier(
    df: pd.DataFrame,
    bucket: str,
    output_dir: Path,
    *,
    x_col: str = "mean_time_seconds",
    x_label: str = "Mean Wall Time (s)",
    log_scale: bool = True,
) -> Path:
    """Generate a single Pareto-front plot for one dimension bucket.

    Args:
        df (pd.DataFrame): DataFrame with trial data for one bucket.
        bucket (str): Bucket name (low/medium/high).
        output_dir (Path): Directory where the figure is saved.
        x_col (str): Column to use as cost objective.
        x_label (str): LaTeX-safe label for the x-axis.
        log_scale (bool): Use log scale on the x-axis.

    Returns:
        Path: Path to the saved figure.
    """
    y_col = "precision_loss"
    y_label = "Mean Relative Error (\\%)"

    pareto_mask = compute_pareto_front(df, x_col=x_col, y_col=y_col)
    pareto_df = df[pareto_mask].sort_values(x_col)

    plt, _ = setup_matplotlib(font_scale=1.3)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    bucket_color = BUCKET_TO_COLOR.get(bucket, "#4c72b0")
    ax.scatter(
        df[x_col].to_numpy(),
        df[y_col].to_numpy(),
        color=bucket_color,
        s=50,
        alpha=0.55,
        edgecolors="gray",
        linewidths=0.5,
        zorder=3,
        label="Trials",
    )

    ax.plot(
        pareto_df[x_col].to_numpy(),
        pareto_df[y_col].to_numpy(),
        color="#C44E52",
        lw=2.5,
        ls="--",
        marker="o",
        markersize=7,
        markerfacecolor="#C44E52",
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Pareto Frontier",
        zorder=4,
    )

    pareto_by_time = pareto_df.sort_values(by=[x_col, y_col])
    low_row = pareto_by_time.iloc[0]

    pareto_losses = np.asarray(pareto_df[y_col].to_numpy(), dtype=float)
    pareto_times = np.log1p(np.asarray(pareto_df[x_col].to_numpy(), dtype=float))

    def minmax(vals: np.ndarray) -> np.ndarray:
        min_v = np.min(vals)
        max_v = np.max(vals)
        if np.isclose(min_v, max_v):
            return np.zeros_like(vals)
        return (vals - min_v) / (max_v - min_v)

    scaled_losses = minmax(pareto_losses)
    scaled_times = minmax(pareto_times)
    balance_scores = np.maximum(scaled_losses, scaled_times)
    euclidean_scores = np.sqrt(scaled_losses**2 + scaled_times**2)
    candidates: list[tuple[float, float, float, float, int, Any]] = []
    for idx_c, (row_idx, row) in enumerate(pareto_df.iterrows()):
        candidates.append(
            (
                float(balance_scores[idx_c]),
                float(euclidean_scores[idx_c]),
                float(row[y_col]),
                float(row[x_col]),
                int(row["trial_number"]),
                row_idx,
            )
        )
    candidates.sort()
    mid_row = pareto_df.loc[candidates[0][-1]]

    pareto_by_loss = pareto_df.sort_values(by=[y_col, x_col])
    high_row = pareto_by_loss.iloc[0]

    ax.scatter(
        low_row[x_col],
        low_row[y_col],
        color="#2ca02c",
        marker="s",
        s=140,
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
        label="Low Budget",
    )
    ax.scatter(
        mid_row[x_col],
        mid_row[y_col],
        color="gold",
        marker="*",
        s=300,
        edgecolors="black",
        linewidths=1.0,
        zorder=7,
        label="Mid Budget",
    )
    ax.scatter(
        high_row[x_col],
        high_row[y_col],
        color="#d62728",
        marker="D",
        s=120,
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
        label="High Budget",
    )

    ax.set_xlabel(x_label, fontsize=FS_LABEL)
    ax.set_ylabel(y_label, fontsize=FS_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax.tick_params(axis="both", which="minor", labelsize=FS_TICK - 2)

    if log_scale:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=10))
        ax.xaxis.set_minor_formatter(NullFormatter())

    ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    path = output_dir / f"pareto_frontier_{bucket}.pdf"
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Pareto plot saved: %s", repo_relative(path))
    return path


def main() -> None:
    """Generate the Pareto-front plots of the hyperparameter tuning (CLI entry point)."""
    parser = argparse.ArgumentParser(description="Generate Pareto-front plots from Optuna tuning results.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to the Optuna results directory. Defaults to the optuna subdir of "
        "the hyperparameter_tuning results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where plots are saved. Defaults to a 'pareto_plots' subdirectory inside the results dir.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="time",
        choices=["time", "evals"],
        help="Cost objective: 'time' (mean_time_seconds) or 'evals' (mean_n_model_evals). Default: 'time'.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log scale on the x-axis.",
    )
    args = parser.parse_args()

    hp_tuning_dir = REPO_ROOT / "experiments" / "hyperparameter_tuning"
    results_dir = Path(args.results_dir) if args.results_dir else (hp_tuning_dir / "results" / "optuna")
    output_dir = Path(args.output_dir) if args.output_dir else (results_dir / "pareto_plots")

    layout = experiment_layout()
    ensure_layout(layout)
    configure_logging(layout, "visualize_and_report")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results dir: %s", repo_relative(results_dir))
    logger.info("Output dir:  %s", repo_relative(output_dir))

    if args.metric == "evals":
        x_col = "mean_n_model_evals"
        x_label = "Mean Number of Model Evaluations"
    else:
        x_col = "mean_time_seconds"
        x_label = "Mean Wall Time (s)"

    log_scale = not args.no_log

    dfs = load_optuna_trials(results_dir)
    if not dfs:
        logger.error("No trial data loaded. Exiting.")
        sys.exit(1)

    paths: list[Path] = []
    for bucket, _, _ in BUCKETS_ORDERED:
        if bucket not in dfs:
            logger.warning("Skipping bucket '%s': no data.", bucket)
            continue
        p = plot_pareto_frontier(
            dfs[bucket],
            bucket,
            output_dir,
            x_col=x_col,
            x_label=x_label,
            log_scale=log_scale,
        )
        paths.append(p)

    logger.info("Done — %d plot(s) saved to %s", len(paths), repo_relative(output_dir))


if __name__ == "__main__":
    main()
