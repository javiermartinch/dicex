# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tables of the computational study: execution times, complexity overhead, and dimensionality scaling."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.log_config import configure_logging, get_logger

DATASET_PRETTY: dict[str, str] = {
    "banknote_authentication": "Banknote",
    "breast_cancer_wisconsin": "Breast Cancer",
    "parkinsons": "Parkinsons",
    "pm10": "PM10",
    "spambase": "Spambase",
    "vehicle_norm": "Vehicle",
    "buzz_toms_hardware": "Buzz Tom's Hardware",
    "cpu_act": "CPU Act",
    "elevators": "Elevators",
    "kin8nm": "Kin8nm",
    "physicochemical_protein": "Protein",
    "superconductivity": "Superconductivity",
}


def generate_base_times_table(
    df: pd.DataFrame,
    tables_dir: Path,
) -> pd.DataFrame:
    """Generate the DiCEx execution-time table (``tab:base-times``) across datasets and model families.

    Args:
        df (pd.DataFrame): Benchmark raw results DataFrame.
        tables_dir (Path): Output directory for paper tables.

    Returns:
        pd.DataFrame: Execution-time summary DataFrame.
    """
    df_dicex = df[df["method"] == "dicex"].copy()
    datasets = sorted(df_dicex["dataset_name"].unique())

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        df_ds = df_dicex[df_dicex["dataset_name"] == dataset]
        if len(df_ds) == 0:
            continue

        n_features = int(df_ds["n_features"].iloc[0])
        task = str(df_ds["task"].iloc[0]).capitalize()

        row: dict[str, Any] = {"Dataset": DATASET_PRETTY.get(dataset, dataset), "d": n_features, "Task": task}

        models_mapping = {
            "Linear": "linear",
            "Logistic": "logistic",
            "Random Forest": "random_forest",
            "MLP": "mlp",
            "XGBoost": "xgboost",
        }

        for col_name, model_family in models_mapping.items():
            df_model = df_ds[df_ds["model_family"] == model_family]
            if len(df_model) > 0:
                mean_t = float(df_model["elapsed_seconds"].mean())
                std_t = float(df_model["elapsed_seconds"].std())
                if pd.isna(std_t):
                    std_t = 0.0

                row[f"{col_name}_Mean"] = round(mean_t, 3)
                row[f"{col_name}_Std"] = round(std_t, 3)
                row[col_name] = f"{mean_t:.2f} ± {std_t:.2f}"
            else:
                row[f"{col_name}_Mean"] = np.nan
                row[f"{col_name}_Std"] = np.nan
                row[col_name] = "N/A"

        rows.append(row)

    res_df = pd.DataFrame(rows)
    csv_path = tables_dir / "base_times.csv"
    res_df.to_csv(csv_path, index=False)
    return res_df


def generate_complexity_overhead_table(
    results_dir: Path,
    tables_dir: Path,
) -> pd.DataFrame:
    """Generate the model-complexity overhead table (``tab:complexity-overhead``).

    Args:
        results_dir (Path): Directory containing raw JSON results.
        tables_dir (Path): Output directory for paper tables.

    Returns:
        pd.DataFrame: Complexity overhead DataFrame.
    """
    complexity_json_path = results_dir / "model_complexity.json"
    if not complexity_json_path.exists():
        return pd.DataFrame()

    with open(complexity_json_path) as f:
        data = json.load(f)

    rows: list[dict[str, Any]] = []
    for item in data:
        family = str(item["model_family"])
        comp_val = str(item["complexity_value"])
        mean_t = float(item["mean_time"])
        std_t = float(item["std_time"])
        mean_evals = float(item["mean_evals"])
        predict_t = float(item["predict_time_sec"])
        overhead = max(0.0, mean_t - predict_t)
        overhead_pct = (overhead / mean_t) * 100.0 if mean_t > 0 else 0.0

        row = {
            "Family": family,
            "Complexity": comp_val,
            "DiCEx Time (s)": f"{mean_t:.3f} ± {std_t:.3f}",
            "Model Inference Time (s)": f"{predict_t:.3f}",
            "Evals": f"{mean_evals:,.0f}",
            "Overhead (%)": f"{overhead_pct:.2f}%",
        }
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    csv_path = tables_dir / "complexity_overhead.csv"
    df_comp.to_csv(csv_path, index=False)
    return df_comp


def generate_dimensionality_scaling_table(
    results_dir: Path,
    tables_dir: Path,
) -> pd.DataFrame:
    """Generate the dimensionality scaling table (``tab:dim-scaling``) across input dimensions d.

    Args:
        results_dir (Path): Directory containing raw JSON results.
        tables_dir (Path): Output directory for paper tables.

    Returns:
        pd.DataFrame: Dimensionality scaling DataFrame.
    """
    dim_json_path = results_dir / "dimensionality_rf.json"
    if not dim_json_path.exists():
        return pd.DataFrame()

    with open(dim_json_path) as f:
        data = json.load(f)

    bench_path = results_dir / "unit_inference_benchmark.csv"
    bench_data: dict[int, tuple[float, float]] = {}
    if bench_path.exists():
        df_bench = pd.read_csv(bench_path)
        for _, b_row in df_bench.iterrows():
            bench_data[int(b_row["d"])] = (
                float(b_row["mean_inference_time_ms_1000_pts"]),
                float(b_row["std_inference_time_ms_1000_pts"]),
            )

    by_d: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in data:
        d = int(run["d"])
        by_d[d].append(run)

    rows: list[dict[str, Any]] = []
    for d in sorted(by_d.keys()):
        runs = by_d[d]

        all_elapsed: list[float] = []
        all_evals: list[float] = []
        all_iters: list[float] = []
        predict_times: list[float] = []

        for run in runs:
            predict_times.append(float(run["mean_predict_time_sec"]))
            for pt in run["points"]:
                all_elapsed.append(float(pt["elapsed_seconds"]))
                all_evals.append(float(pt["n_model_evals"]))
                all_iters.append(float(pt["n_iterations"]))

        mean_dicex = float(np.mean(all_elapsed))
        std_dicex = float(np.std(all_elapsed))
        mean_infer = float(np.mean(predict_times))

        overhead_pct = ((mean_dicex - mean_infer) / mean_dicex) * 100.0 if mean_dicex > 0 else 0.0

        unit_inf_str = "N/A"
        if d in bench_data:
            mean_b, std_b = bench_data[d]
            unit_inf_str = f"{mean_b:.2f} ± {std_b:.2f} ms"

        row = {
            "d": d,
            "DiCEx Time (s)": f"{mean_dicex:.2f} ± {std_dicex:.2f}",
            "Inference (s)": f"{mean_infer:.3f}",
            "Unit Inference (1000 pts)": unit_inf_str,
            "Overhead (%)": f"{overhead_pct:.1f}%",
            "n_evals": f"{np.mean(all_evals):,.0f}",
            "n_iters": f"{np.mean(all_iters):.1f}",
        }
        rows.append(row)

    df_dim = pd.DataFrame(rows)
    csv_path = tables_dir / "dimensionality_scaling.csv"
    df_dim.to_csv(csv_path, index=False)
    return df_dim


def main() -> None:
    """Generate the computational study tables (CLI entry point)."""
    parser = argparse.ArgumentParser(description="Generate paper tables for computational study.")
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to comparative audit results CSV.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    tables_dir = results_dir / "tables" / "paper"
    log_dir = results_dir / "logs"

    configure_logging(script_name="visualize_and_report", log_dir=log_dir)
    logger = get_logger("computational_study.visualize_and_report")

    tables_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        csv_file = Path(args.input_file)
    else:
        csv_file = REPO_ROOT / "experiments" / "comparative_audit" / "results" / "paper_results.csv"

    if not csv_file.exists():
        logger.error("Comparative audit results file not found: %s", csv_file)
        sys.exit(1)

    logger.info("Loading results from %s", csv_file)
    df = pd.read_csv(csv_file)

    logger.info("Generating computational study summary tables...")
    t1_df = generate_base_times_table(df, tables_dir)
    logger.info("Execution times (tab:base-times):\n%s", t1_df.to_string(index=False))

    t2_df = generate_complexity_overhead_table(results_dir, tables_dir)
    logger.info("Model complexity overhead (tab:complexity-overhead):\n%s", t2_df.to_string(index=False))

    t3_df = generate_dimensionality_scaling_table(results_dir, tables_dir)
    logger.info("Dimensionality scaling (tab:dim-scaling):\n%s", t3_df.to_string(index=False))

    logger.info("Computational study tables generated.")


if __name__ == "__main__":
    main()
