# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Figures and tables of the comparative audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.persistence import ensure_layout, experiment_layout
from experiments.shared.plots import setup_latex_matplotlib

logger = get_logger("comparative_audit.visualize_and_report")

METHOD_ORDER: list[str] = ["dicex", "lime", "nominal", "random_direction"]
METHOD_PRETTY: dict[str, str] = {
    "dicex": "DiCEx",
    "lime": "LIME-derived",
    "nominal": "Nominal",
    "random_direction": "Random Dir.",
}


def plot_advantage_heatmaps(df: pd.DataFrame, plots_dir: Path) -> dict[str, Path]:
    """Generate the three individual advantage heatmaps for non-linear models.

    Args:
        df (pd.DataFrame): Benchmark results dataframe.
        plots_dir (Path): Output directory for plot files.

    Returns:
        dict[str, Path]: Map of plot names to saved file paths.
    """
    plt, sns = setup_latex_matplotlib(font_scale=1.1)

    model_families = ["mlp", "random_forest", "xgboost"]
    plot_df = df[df["model_family"].isin(model_families)].copy()
    plot_df = plot_df[plot_df["method"].str.lower().isin(["dicex", "lime", "nominal"])].copy()
    plot_df["method"] = plot_df["method"].str.lower()

    grouped = (
        plot_df.groupby(["noise_label", "alpha", "method"], observed=True)
        .agg(
            failure_rate=("failure_rate", "mean"),
            cvar_norm_maxmin=("cvar_norm_maxmin", "mean"),
            volatility_norm_maxmin=("volatility_norm_maxmin", "mean"),
        )
        .reset_index()
    )

    pivot_cvar = grouped.pivot_table(
        index=["alpha", "noise_label"], columns="method", values="cvar_norm_maxmin", observed=True
    )
    pivot_fr = grouped.pivot_table(
        index=["alpha", "noise_label"], columns="method", values="failure_rate", observed=True
    )
    pivot_vol = grouped.pivot_table(
        index=["alpha", "noise_label"], columns="method", values="volatility_norm_maxmin", observed=True
    )

    for pivot in [pivot_cvar, pivot_fr, pivot_vol]:
        for m in ["dicex", "lime", "nominal"]:
            if m not in pivot.columns:
                pivot[m] = np.nan
        for col in ["lime", "nominal"]:
            pivot[col] = pivot.groupby(level="noise_label")[col].transform(lambda x: x.ffill().bfill())

    # Absolute metric calculations
    delta_cvar = pivot_cvar["dicex"] - np.maximum(pivot_cvar["lime"], pivot_cvar["nominal"])

    best_fr = np.minimum(pivot_fr["lime"], pivot_fr["nominal"])
    delta_fr = (best_fr - pivot_fr["dicex"]) * 100

    best_vol = np.minimum(pivot_vol["lime"], pivot_vol["nominal"])
    delta_vol = best_vol - pivot_vol["dicex"]

    noise_order = ["low_noise", "balanced", "elevated_noise"]
    noise_cols = {
        "low_noise": "Low\n(0.05)",
        "balanced": "Balanced\n(0.25)",
        "elevated_noise": "Elevated\n(0.35)",
    }

    def to_matrix(series: pd.Series) -> pd.DataFrame:
        mat = series.unstack(level="noise_label")  # noqa: PD010
        mat = mat.sort_index(ascending=True)
        mat = mat.reindex(columns=noise_order)
        mat.columns = [noise_cols[c] for c in mat.columns]
        return mat

    mat_cvar = to_matrix(delta_cvar)
    mat_fr = to_matrix(delta_fr)
    mat_vol = to_matrix(delta_vol)

    # File names match the figures included in the paper (Fig. heatmaps-noise-alpha).
    heatmaps = [
        (
            "heatmap_cvar_gain",
            mat_cvar,
            lambda v: f"+{v:.4f}" if v > 0 else f"{v:.4f}",
        ),
        (
            "heatmap_fail_rate_reduction",
            mat_fr,
            lambda v: f"+{v:.1f}\\%" if v > 0 else f"{v:.1f}\\%",
        ),
        (
            "heatmap_volatility_reduction",
            mat_vol,
            lambda v: f"+{v:.4f}" if v > 0 else f"{v:.4f}",
        ),
    ]

    saved_paths = {}
    old_savefig_bbox = plt.rcParams.get("savefig.bbox", None)
    plt.rcParams["savefig.bbox"] = None
    for name, mat, annot_fn in heatmaps:
        fig, ax = plt.subplots(figsize=(9.0, 6.5))
        cax = fig.add_axes((0.78, 0.16, 0.04, 0.76))
        ax.set_position((0.15, 0.16, 0.60, 0.76))

        vmax = max(abs(float(mat.min().min())), abs(float(mat.max().max())))
        if vmax < 1e-8:
            vmax = 1.0
        vmin = -vmax

        sns.heatmap(
            mat,
            annot=mat.map(annot_fn),
            fmt="",
            cmap="RdBu",
            center=0,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            cbar=True,
            cbar_ax=cax,
            ax=ax,
            annot_kws={"size": 22},
        )

        ax.set_xlabel(r"$\sigma$", fontsize=38, labelpad=10)
        ax.set_ylabel(r"$\alpha$", fontsize=38, labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=26)
        cax.tick_params(labelsize=26)

        pdf_path = plots_dir / f"{name}.pdf"
        fig.savefig(pdf_path, dpi=300)
        plt.close(fig)

        saved_paths[name] = pdf_path
        logger.info("Saved heatmap %s -> %s", name, pdf_path.name)
    plt.rcParams["savefig.bbox"] = old_savefig_bbox

    return saved_paths


def plot_pareto_frontier(df: pd.DataFrame, plots_dir: Path) -> Path:
    """Generate the risk-reward Pareto frontier plot comparing methods.

    Args:
        df (pd.DataFrame): Benchmark results dataframe.
        plots_dir (Path): Output directory for plot files.

    Returns:
        Path: Path to saved PDF plot.
    """
    plt, sns = setup_latex_matplotlib(font_scale=1.1)

    plot_df = df.copy()

    def get_group_name(row: pd.Series) -> str:
        pretty_name = METHOD_PRETTY.get(str(row["method"]), str(row["method"]))
        if row["method"] == "dicex":
            return f"{pretty_name} ($\\alpha={float(row['alpha']):.2f}$)"
        return pretty_name

    plot_df["method_group"] = plot_df.apply(get_group_name, axis=1)

    centroids = (
        plot_df.groupby("method_group", observed=True)
        .agg(
            mean_volatility=("volatility_norm_maxmin", "mean"),
            mean_gain=("mean_gain_norm_maxmin", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    alpha_methods = ["DiCEx"]
    fixed_methods = ["Nominal", "LIME-derived", "Random Dir."]
    present_groups = [m for m in fixed_methods if m in plot_df["method_group"].unique()]
    for pretty_base in alpha_methods:
        alpha_groups = sorted(
            [g for g in plot_df["method_group"].unique() if g.startswith(pretty_base)],
            key=lambda x: float(x.split("=")[1].replace("$", "").replace(")", "")),
        )
        present_groups.extend(alpha_groups)

    palette = sns.color_palette("husl", len(present_groups))
    color_map = dict(zip(present_groups, palette, strict=False))

    markers = {}
    for g in present_groups:
        if g.startswith("DiCEx"):
            markers[g] = "*"
        elif g == "Nominal":
            markers[g] = "D"
        elif g == "LIME-derived":
            markers[g] = "o"
        else:
            markers[g] = "s"

    centroids = centroids[centroids["method_group"].isin(present_groups)].copy()

    for _, row in centroids.iterrows():
        group = row["method_group"]
        if group not in present_groups:
            continue
        marker = markers.get(group, "o")
        size = 280 if marker == "*" else 160

        ax.scatter(
            row["mean_volatility"],
            row["mean_gain"],
            color="white",
            s=size + 100,
            edgecolors="none",
            marker=marker,
            zorder=9,
        )

        ax.scatter(
            row["mean_volatility"],
            row["mean_gain"],
            color=color_map.get(group, "gray"),
            s=size,
            edgecolors="black",
            linewidth=1.2,
            marker=marker,
            label=group,
            zorder=10,
        )

    dicex_centroids = centroids[centroids["method_group"].str.startswith("DiCEx")]
    if len(dicex_centroids) > 1:
        dicex_centroids = dicex_centroids.sort_values("mean_volatility")
        ax.plot(
            dicex_centroids["mean_volatility"],
            dicex_centroids["mean_gain"],
            color="black",
            linestyle="-",
            linewidth=2,
            alpha=0.5,
            zorder=8,
        )

    # Add alpha tags to DiCEx points
    tag_config = {
        0.10: {"dx": 0.0, "dy": 0.0055, "ha": "center", "va": "bottom"},
        0.25: {"dx": 0.0, "dy": 0.0055, "ha": "center", "va": "bottom"},
        0.50: {"dx": 0.0, "dy": 0.0055, "ha": "center", "va": "bottom"},
        0.75: {"dx": 0.0005, "dy": 0.0055, "ha": "right", "va": "bottom"},
        0.90: {"dx": -0.0005, "dy": 0.0055, "ha": "left", "va": "bottom"},
    }

    for _, row in dicex_centroids.iterrows():
        group = row["method_group"]
        alpha_str = group.split("=")[1].replace("$", "").replace(")", "")
        alpha_val = round(float(alpha_str), 2)
        cfg = tag_config.get(alpha_val, {"dx": 0.0, "dy": 0.0025, "ha": "center", "va": "bottom"})
        color = color_map.get(group, "#333333")
        ax.text(
            row["mean_volatility"] + cfg["dx"],
            row["mean_gain"] + cfg["dy"],
            rf"$\alpha = {alpha_val:.2f}$",
            fontsize=12,
            ha=cfg["ha"],
            va=cfg["va"],
            zorder=12,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec=color,
                lw=1.4,
                alpha=0.95,
            ),
        )

    fs_title, fs_label, fs_legend, fs_annot = 22, 18, 16, 14
    ax.set_xlabel("Risk (Normalized Volatility)", fontsize=fs_label)
    ax.set_ylabel(r"Reward ($\widetilde{\Delta}_1$)", fontsize=fs_label)
    ax.tick_params(axis="both", which="major", labelsize=fs_annot)
    ax.set_title("Risk-Reward Pareto Frontier", fontsize=fs_title, pad=15)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.3, color="0.35")

    ax.legend(loc="lower right", fontsize=fs_legend, frameon=True, framealpha=0.85)

    vol_values = centroids["mean_volatility"]
    gain_values = centroids["mean_gain"]
    vol_min, vol_max = vol_values.min(), vol_values.max()
    gain_min, gain_max = gain_values.min(), gain_values.max()
    vol_range = vol_max - vol_min if vol_max > vol_min else 0.1
    gain_range = gain_max - gain_min if gain_max > gain_min else 0.1
    ax.set_xlim(max(0, vol_min - vol_range * 0.15), vol_max + vol_range * 0.15)
    ax.set_ylim(gain_min - gain_range * 0.15, gain_max + gain_range * 0.22)

    fig.tight_layout()
    pdf_path = plots_dir / "pareto_frontier.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info("Saved Pareto plot -> %s", pdf_path.name)
    return pdf_path


def generate_grand_summary_table(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Generate the overall comparison table (``tab:audit-grand-summary``).

    Args:
        df (pd.DataFrame): Filtered benchmark results dataframe.
        tables_dir (Path): Output directory for table files.

    Returns:
        pd.DataFrame: Table dataframe.
    """
    grand_rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        if len(sub) == 0:
            continue
        grand_rows.append(
            {
                "Method": METHOD_PRETTY[method],
                "Norm. CVaR_mean": sub["cvar_norm_maxmin"].mean(),
                "Norm. CVaR_std": sub["cvar_norm_maxmin"].std(),
                "Norm. Mean Gain_mean": sub["mean_gain_norm_maxmin"].mean(),
                "Norm. Mean Gain_std": sub["mean_gain_norm_maxmin"].std(),
                "Failure Rate (%)_mean": sub["failure_rate"].mean() * 100,
                "Failure Rate (%)_std": sub["failure_rate"].std() * 100,
                "Norm. Volatility_mean": sub["volatility_norm_maxmin"].mean(),
                "Norm. Volatility_std": sub["volatility_norm_maxmin"].std(),
            }
        )
    df_raw = pd.DataFrame(grand_rows)

    csv_df = pd.DataFrame(
        {
            "Method": df_raw["Method"],
            "Norm. CVaR": df_raw.apply(lambda r: f"{r['Norm. CVaR_mean']:.4f} ± {r['Norm. CVaR_std']:.4f}", axis=1),
            "Norm. Mean Gain": df_raw.apply(
                lambda r: f"{r['Norm. Mean Gain_mean']:.4f} ± {r['Norm. Mean Gain_std']:.4f}", axis=1
            ),
            "Failure Rate (%)": df_raw.apply(
                lambda r: f"{r['Failure Rate (%)_mean']:.2f} ± {r['Failure Rate (%)_std']:.2f}", axis=1
            ),
            "Norm. Volatility": df_raw.apply(
                lambda r: f"{r['Norm. Volatility_mean']:.4f} ± {r['Norm. Volatility_std']:.4f}", axis=1
            ),
        }
    )

    csv_path = tables_dir / "audit_grand_summary.csv"
    csv_df.to_csv(csv_path, index=False)

    return csv_df


def generate_model_type_split_table(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Generate the comparison table by model complexity (``tab:audit-model-type-split``).

    Args:
        df (pd.DataFrame): Filtered benchmark results dataframe.
        tables_dir (Path): Output directory for table files.

    Returns:
        pd.DataFrame: Table dataframe.
    """
    df_linear = df[df["model_family"].isin(["linear", "logistic"])]
    df_nonlinear = df[df["model_family"].isin(["random_forest", "xgboost", "mlp"])]

    panel_a_rows, panel_b_rows = [], []
    for method in METHOD_ORDER:
        sub_a = df_linear[df_linear["method"] == method]
        if len(sub_a) > 0:
            panel_a_rows.append(
                {
                    "Method": METHOD_PRETTY[method],
                    "Norm. CVaR_mean": sub_a["cvar_norm_maxmin"].mean(),
                    "Norm. CVaR_std": sub_a["cvar_norm_maxmin"].std(),
                    "Norm. Mean Gain_mean": sub_a["mean_gain_norm_maxmin"].mean(),
                    "Norm. Mean Gain_std": sub_a["mean_gain_norm_maxmin"].std(),
                    "Failure Rate (%)_mean": sub_a["failure_rate"].mean() * 100,
                    "Failure Rate (%)_std": sub_a["failure_rate"].std() * 100,
                    "Norm. Volatility_mean": sub_a["volatility_norm_maxmin"].mean(),
                    "Norm. Volatility_std": sub_a["volatility_norm_maxmin"].std(),
                }
            )

        sub_b = df_nonlinear[df_nonlinear["method"] == method]
        if len(sub_b) > 0:
            panel_b_rows.append(
                {
                    "Method": METHOD_PRETTY[method],
                    "Norm. CVaR_mean": sub_b["cvar_norm_maxmin"].mean(),
                    "Norm. CVaR_std": sub_b["cvar_norm_maxmin"].std(),
                    "Norm. Mean Gain_mean": sub_b["mean_gain_norm_maxmin"].mean(),
                    "Norm. Mean Gain_std": sub_b["mean_gain_norm_maxmin"].std(),
                    "Failure Rate (%)_mean": sub_b["failure_rate"].mean() * 100,
                    "Failure Rate (%)_std": sub_b["failure_rate"].std() * 100,
                    "Norm. Volatility_mean": sub_b["volatility_norm_maxmin"].mean(),
                    "Norm. Volatility_std": sub_b["volatility_norm_maxmin"].std(),
                }
            )

    df_raw_a = pd.DataFrame(panel_a_rows)
    df_raw_b = pd.DataFrame(panel_b_rows)

    t2_a = pd.DataFrame(
        {
            "Method": df_raw_a["Method"],
            "Norm. CVaR": df_raw_a.apply(lambda r: f"{r['Norm. CVaR_mean']:.4f} ± {r['Norm. CVaR_std']:.4f}", axis=1),
            "Norm. Mean Gain": df_raw_a.apply(
                lambda r: f"{r['Norm. Mean Gain_mean']:.4f} ± {r['Norm. Mean Gain_std']:.4f}", axis=1
            ),
            "Failure Rate (%)": df_raw_a.apply(
                lambda r: f"{r['Failure Rate (%)_mean']:.2f} ± {r['Failure Rate (%)_std']:.2f}", axis=1
            ),
            "Norm. Volatility": df_raw_a.apply(
                lambda r: f"{r['Norm. Volatility_mean']:.4f} ± {r['Norm. Volatility_std']:.4f}", axis=1
            ),
        }
    )
    t2_b = pd.DataFrame(
        {
            "Method": df_raw_b["Method"],
            "Norm. CVaR": df_raw_b.apply(lambda r: f"{r['Norm. CVaR_mean']:.4f} ± {r['Norm. CVaR_std']:.4f}", axis=1),
            "Norm. Mean Gain": df_raw_b.apply(
                lambda r: f"{r['Norm. Mean Gain_mean']:.4f} ± {r['Norm. Mean Gain_std']:.4f}", axis=1
            ),
            "Failure Rate (%)": df_raw_b.apply(
                lambda r: f"{r['Failure Rate (%)_mean']:.2f} ± {r['Failure Rate (%)_std']:.2f}", axis=1
            ),
            "Norm. Volatility": df_raw_b.apply(
                lambda r: f"{r['Norm. Volatility_mean']:.4f} ± {r['Norm. Volatility_std']:.4f}", axis=1
            ),
        }
    )

    t2_combined = pd.concat(
        [
            pd.DataFrame([{"Method": "Panel A: Linear and Logistic Models"}]),
            t2_a,
            pd.DataFrame([{"Method": "Panel B: Complex Non-linear Models"}]),
            t2_b,
        ],
        ignore_index=True,
    )

    csv_path = tables_dir / "audit_model_type_split.csv"
    t2_combined.to_csv(csv_path, index=False)

    return t2_combined


def generate_noise_breakdown_table(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Generate the comparison table by noise level on complex models (``tab:noise-breakdown``).

    Args:
        df (pd.DataFrame): Filtered benchmark results dataframe.
        tables_dir (Path): Output directory for table files.

    Returns:
        pd.DataFrame: Table dataframe.
    """
    df_nonlinear = df[df["model_family"].isin(["random_forest", "xgboost", "mlp"])]

    noise_rows = []
    for method in METHOD_ORDER:
        sub_method = df_nonlinear[df_nonlinear["method"] == method]
        if len(sub_method) == 0:
            continue

        row = {"Method": METHOD_PRETTY[method]}
        for noise in ["low_noise", "balanced", "elevated_noise"]:
            sub_noise = sub_method[sub_method["noise_label"] == noise]
            lbl = {"low_noise": "Low", "balanced": "Balanced", "elevated_noise": "Elevated"}[noise]

            row[f"CVaR_{lbl}_mean"] = sub_noise["cvar_norm_maxmin"].mean()
            row[f"CVaR_{lbl}_std"] = sub_noise["cvar_norm_maxmin"].std()
            row[f"Failure_{lbl}_mean"] = sub_noise["failure_rate"].mean() * 100
            row[f"Failure_{lbl}_std"] = sub_noise["failure_rate"].std() * 100
            row[f"Volatility_{lbl}_mean"] = sub_noise["volatility_norm_maxmin"].mean()
            row[f"Volatility_{lbl}_std"] = sub_noise["volatility_norm_maxmin"].std()

        noise_rows.append(row)

    df_raw = pd.DataFrame(noise_rows)

    t3 = pd.DataFrame({"Method": df_raw["Method"]})
    for lbl in ["Low", "Balanced", "Elevated"]:
        t3[f"CVaR_{lbl}"] = [
            f"{m:.4f} ± {s:.4f}" for m, s in zip(df_raw[f"CVaR_{lbl}_mean"], df_raw[f"CVaR_{lbl}_std"], strict=False)
        ]
        t3[f"Failure_{lbl}"] = [
            f"{m:.2f} ± {s:.2f}"
            for m, s in zip(df_raw[f"Failure_{lbl}_mean"], df_raw[f"Failure_{lbl}_std"], strict=False)
        ]
        t3[f"Volatility_{lbl}"] = [
            f"{m:.4f} ± {s:.4f}"
            for m, s in zip(df_raw[f"Volatility_{lbl}_mean"], df_raw[f"Volatility_{lbl}_std"], strict=False)
        ]

    csv_path = tables_dir / "audit_noise_breakdown.csv"
    t3.to_csv(csv_path, index=False)

    return t3


def generate_percentile_table(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Generate the comparison table by query percentile on complex models (``tab:audit-percentile``).

    Args:
        df (pd.DataFrame): Filtered benchmark results dataframe.
        tables_dir (Path): Output directory for table files.

    Returns:
        pd.DataFrame: Table dataframe.
    """
    df_nonlinear = df[df["model_family"].isin(["random_forest", "xgboost", "mlp"])]

    percentile_rows = []
    for method in METHOD_ORDER:
        sub_method = df_nonlinear[df_nonlinear["method"] == method]
        if len(sub_method) == 0:
            continue

        row = {"Method": METHOD_PRETTY[method]}
        for perc in [0.1, 0.5, 0.9]:
            sub_perc = sub_method[np.isclose(sub_method["query_percentile"], perc)]
            lbl = {0.1: "10th", 0.5: "50th", 0.9: "90th"}[perc]

            row[f"CVaR_{lbl}_mean"] = sub_perc["cvar_norm_maxmin"].mean()
            row[f"CVaR_{lbl}_std"] = sub_perc["cvar_norm_maxmin"].std()
            row[f"Failure_{lbl}_mean"] = sub_perc["failure_rate"].mean() * 100
            row[f"Failure_{lbl}_std"] = sub_perc["failure_rate"].std() * 100

        percentile_rows.append(row)

    df_raw = pd.DataFrame(percentile_rows)

    t4 = pd.DataFrame({"Method": df_raw["Method"]})
    for lbl in ["10th", "50th", "90th"]:
        t4[f"CVaR_{lbl}"] = [
            f"{m:.4f} ± {s:.4f}" for m, s in zip(df_raw[f"CVaR_{lbl}_mean"], df_raw[f"CVaR_{lbl}_std"], strict=False)
        ]
        t4[f"Failure_{lbl}"] = [
            f"{m:.2f} ± {s:.2f}"
            for m, s in zip(df_raw[f"Failure_{lbl}_mean"], df_raw[f"Failure_{lbl}_std"], strict=False)
        ]

    csv_path = tables_dir / "audit_percentile.csv"
    t4.to_csv(csv_path, index=False)

    return t4


def main() -> None:
    """Generate the comparative audit plots and tables (CLI entry point)."""
    parser = argparse.ArgumentParser(description="Generate plots and tables for comparative audit.")
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to results CSV file.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    plots_dir = results_dir / "plots" / "paper"
    tables_dir = results_dir / "tables" / "paper"

    layout = experiment_layout(results_root=results_dir)
    ensure_layout(layout)
    configure_logging(layout, "visualize_and_report")

    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        csv_file = Path(args.input_file)
    else:
        csv_file = results_dir / "paper_results.csv"

    if not csv_file.exists():
        logger.error("Results file not found: %s", csv_file)
        sys.exit(1)

    logger.info("Loading results from %s", csv_file)
    df = pd.read_csv(csv_file)

    logger.info("Generating evaluation plots...")
    plot_advantage_heatmaps(df, plots_dir)
    plot_pareto_frontier(df, plots_dir)

    logger.info("Generating summary tables...")
    t1_df = generate_grand_summary_table(df, tables_dir)
    logger.info("Overall comparison table (tab:audit-grand-summary):\n%s", t1_df.to_string(index=False))

    t2_df = generate_model_type_split_table(df, tables_dir)
    logger.info("Comparison by model complexity (tab:audit-model-type-split):\n%s", t2_df.to_string(index=False))

    t3_df = generate_noise_breakdown_table(df, tables_dir)
    logger.info("Comparison by noise level (tab:noise-breakdown):\n%s", t3_df.to_string(index=False))

    t4_df = generate_percentile_table(df, tables_dir)
    logger.info("Comparison by query percentile (tab:audit-percentile):\n%s", t4_df.to_string(index=False))

    logger.info("Comparative audit plots and tables generated.")


if __name__ == "__main__":
    main()
