# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Figures and tables of the analytic convergence study."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.persistence import ensure_layout, experiment_layout
from experiments.shared.plots import setup_latex_matplotlib

logger = get_logger("visualize_and_report")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PAPER_DIR = RESULTS_DIR / "plots" / "paper"
BOXPLOTS_DIR = PAPER_DIR / "boxplots"
TABLES_DIR = PAPER_DIR / "tables"

PRESET_SIGNATURES: dict[str, dict[str, int]] = {
    "minimal": {"b_samples": 8, "n_exploration": 50, "n_test": 300, "n_evaluation": 4000, "n_bootstrap": 300},
    "low": {"b_samples": 16, "n_exploration": 200, "n_test": 800, "n_evaluation": 7500, "n_bootstrap": 800},
    "mid": {"b_samples": 32, "n_exploration": 500, "n_test": 1500, "n_evaluation": 10000, "n_bootstrap": 1500},
    "high": {"b_samples": 64, "n_exploration": 1000, "n_test": 2000, "n_evaluation": 15000, "n_bootstrap": 2000},
    "xhigh": {"b_samples": 256, "n_exploration": 3000, "n_test": 5000, "n_evaluation": 20000, "n_bootstrap": 5000},
}

PRESETS_ORDERED: list[tuple[str, str]] = [
    ("minimal", "Minimal"),
    ("low", "Low"),
    ("mid", "Mid"),
    ("high", "High"),
    ("xhigh", "Extra High"),
]

PRESET_COLORS = ["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3", "#a6d854"]


def assign_preset(df: pd.DataFrame) -> pd.Series:
    """Assign preset labels based on configuration columns.

    Args:
        df (pd.DataFrame): Experiment results DataFrame.

    Returns:
        pd.Series: Series containing assigned preset names.
    """
    preset = pd.Series("unknown", index=df.index, dtype=str)
    for name, sig in PRESET_SIGNATURES.items():
        mask = pd.Series(True, index=df.index)
        for col, val in sig.items():
            if col in df.columns:
                mask &= df[col] == val
        preset[mask] = name
    return preset


def generate_global_boxplot(
    df: pd.DataFrame,
    column: str,
    output_pdf_path: Path,
    ylabel: str,
    scale: float = 1.0,
) -> None:
    """Generate the global boxplot of a given error metric.

    Args:
        df (pd.DataFrame): Results DataFrame.
        column (str): Metric column name to plot.
        output_pdf_path (Path): Path to output PDF file.
        ylabel (str): Y-axis label.
        scale (float): Scaling factor for the metric. Defaults to 1.0.
    """
    plt, _ = setup_latex_matplotlib(font_scale=2.2)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data: list[pd.Series] = []
    labels: list[str] = []
    positions: list[int] = []
    medians: list[float] = []

    for i, (key, short_label) in enumerate(PRESETS_ORDERED):
        vals = df[df["preset"] == key][column] * scale
        data.append(vals)
        labels.append(short_label)
        positions.append(i)
        medians.append(float(vals.median()))

    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, whis=[5, 95], showfliers=False)

    for patch, color in zip(bp["boxes"], PRESET_COLORS, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.plot(positions, medians, color="#333333", ls="--", lw=1.5, marker="o", markersize=7, label="Median")

    ax.set_xticklabels(labels, fontsize=25)
    ax.set_xlabel("Optimizer Preset", fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.tick_params(axis="both", which="major", labelsize=25)
    ax.legend(fontsize=25, loc="upper right", frameon=True, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def generate_presets_table(output_dir: Path) -> None:
    """Generate the LaTeX and CSV table of the optimizer preset parameters.

    Args:
        output_dir (Path): Output directory for saving table files.
    """
    config_path = BASE_DIR / "experiment_config.yaml"
    presets_data = []

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        preset_cfg = cfg.get("optimizer", {}).get("presets", {})
    else:
        preset_cfg = {}

    for key, label in PRESETS_ORDERED:
        p_info = preset_cfg.get(key, {})
        schedule = p_info.get("angular_schedule_deg", [])
        schedule_str = "{" + ", ".join(f"{deg:.0f}^\\circ" for deg in schedule) + "}"
        presets_data.append(
            {
                "Preset": label,
                "$\\rho_l$": schedule_str,
                "$\\eta$": p_info.get("eta", 0.05),
                "$|\\mathcal{T}|$": p_info.get("grid_t_len", PRESET_SIGNATURES[key]["b_samples"]),
                "$B$": PRESET_SIGNATURES[key]["b_samples"],
                "$n$": PRESET_SIGNATURES[key]["n_exploration"],
                "$n_{\\mathrm{val}}$": PRESET_SIGNATURES[key]["n_test"],
                "$n_{\\mathrm{eval}}$": PRESET_SIGNATURES[key]["n_evaluation"],
                "$B_{\\mathrm{boot}}$": PRESET_SIGNATURES[key]["n_bootstrap"],
            }
        )

    df_presets = pd.DataFrame(presets_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "tab_presets.csv"
    df_presets.to_csv(csv_path, index=False)


def generate_convergence_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate the summary statistics table of angular and relative value errors.

    Args:
        df (pd.DataFrame): Cleaned results DataFrame.
        output_dir (Path): Output directory for saving table files.
    """
    rows = []
    for task_name in ["classification", "regression"]:
        task_df = df[df["task"] == task_name]
        for key, preset_label in PRESETS_ORDERED:
            p_df = task_df[task_df["preset"] == key]
            if len(p_df) == 0:
                continue

            ang = p_df["angle_diff_sol_norm_deg"]
            val = p_df["rel_val_diff_direction"] * 100

            rows.append(
                {
                    "Task": task_name.title(),
                    "Preset": preset_label,
                    "ang_mean": ang.mean(),
                    "ang_std": ang.std(),
                    "ang_median": ang.median(),
                    "ang_q1": ang.quantile(0.25),
                    "ang_q3": ang.quantile(0.75),
                    "val_mean": val.mean(),
                    "val_std": val.std(),
                    "val_median": val.median(),
                    "val_q1": val.quantile(0.25),
                    "val_q3": val.quantile(0.75),
                }
            )

    stats_df = pd.DataFrame(rows).round(2)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "tab_convergence_summary.csv"
    stats_df.to_csv(csv_path, index=False)


def generate_value_regression_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Regress normalized computed values on the normalized optima, per task and preset.

    Fits ``dicex = slope * analytic + intercept`` on the values normalized by the
    prediction range (the diagnostic reported in Section 7.2) and writes the slope,
    intercept, and R^2 to ``tab_value_regression.csv``.

    Args:
        df (pd.DataFrame): Raw convergence results.
        output_dir (Path): Directory where the CSV table is written.
    """
    rows = []
    for (task, preset), group in df.groupby(["task", "preset"]):
        x = group["analytic_cvar_norm_maxmin"].to_numpy(dtype=float)
        y = group["dicex_cvar_norm_maxmin"].to_numpy(dtype=float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]
        slope, intercept = np.polyfit(x, y, deg=1)
        residuals = y - (slope * x + intercept)
        r2 = 1.0 - float(np.sum(residuals**2)) / float(np.sum((y - y.mean()) ** 2))
        rows.append(
            {"Task": task, "Preset": preset, "slope": slope, "intercept": intercept, "r2": r2, "n": int(mask.sum())}
        )
    csv_path = output_dir / "tab_value_regression.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")


def main() -> None:
    """Generate the analytic convergence boxplots and tables (CLI entry point)."""
    layout = experiment_layout(results_root=RESULTS_DIR)
    ensure_layout(layout)
    log_dir = RESULTS_DIR / "logs"
    configure_logging(layout, "visualize_and_report", log_dir=log_dir)

    raw_files = sorted(RESULTS_DIR.glob("raw_results_*.csv"))
    if not raw_files:
        msg = f"No raw_results_*.csv files found in {RESULTS_DIR}"
        raise FileNotFoundError(msg)
    results_csv = raw_files[-1]

    df = pd.read_csv(results_csv)
    df["preset"] = assign_preset(df)

    generate_global_boxplot(
        df=df,
        column="angle_diff_sol_norm_deg",
        output_pdf_path=BOXPLOTS_DIR / "convergence_angular_error.pdf",
        ylabel=r"$\theta_{\mathrm{err}} \; (^\circ)$",
        scale=1.0,
    )

    generate_global_boxplot(
        df=df,
        column="rel_val_diff_direction",
        output_pdf_path=BOXPLOTS_DIR / "convergence_relative_error.pdf",
        ylabel=r"$\varepsilon_{\mathrm{rel}} \; (\%)$",
        scale=100.0,
    )

    generate_presets_table(TABLES_DIR)
    generate_convergence_summary_table(df, TABLES_DIR)
    generate_value_regression_table(df, TABLES_DIR)


if __name__ == "__main__":
    main()
