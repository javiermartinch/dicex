# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Benchmark the prediction time of a random forest on 1000 points across input dimensions."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

if TYPE_CHECKING:
    from numpy.typing import NDArray

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

from experiments.shared.log_config import configure_logging, get_logger

configure_logging(script_name="benchmark_unit_inference", log_dir=LOGS_DIR)
logger = get_logger("computational_study.benchmark_unit_inference")

DIMS = [4, 8, 16, 32, 64, 128, 256, 512]
N_TRAIN = 2000
N_BENCHMARK = 1000
N_TRIALS = 10


def generate_synthetic_data(
    d: int,
    n_samples: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a synthetic linear dataset with Gaussian features and target noise.

    Args:
        d (int): Number of feature dimensions.
        n_samples (int): Number of sample observations to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Feature matrix and target vector.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n_samples, d)).astype(np.float64)
    beta = np.ones(d, dtype=np.float64) / np.sqrt(d)
    y = x @ beta + 0.01 * rng.normal(0, 1, size=n_samples).astype(np.float64)
    return x, y


def main() -> None:
    """Run the unit-inference benchmark across feature dimensions (CLI entry point)."""
    logger.info("Benchmarking unit inference for 1000 points across %d trials...", N_TRIALS)

    rows: list[dict[str, object]] = []
    for d in DIMS:
        x_train, y_train = generate_synthetic_data(d, N_TRAIN, seed=42)
        x_test, _ = generate_synthetic_data(d, N_BENCHMARK, seed=123)

        model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        model.fit(x_train, y_train)

        model.predict(x_test)

        times: list[float] = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            model.predict(x_test)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        mean_time_ms = float(np.mean(times)) * 1000.0
        std_time_ms = float(np.std(times)) * 1000.0

        logger.info("%5d | %33.3f ± %.3f ms", d, mean_time_ms, std_time_ms)

        rows.append(
            {
                "d": d,
                "mean_inference_time_ms_1000_pts": round(mean_time_ms, 3),
                "std_inference_time_ms_1000_pts": round(std_time_ms, 3),
            }
        )

    df_res = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "unit_inference_benchmark.csv"
    df_res.to_csv(out_csv, index=False)
    logger.info("Saved benchmark results to %s", out_csv)


if __name__ == "__main__":
    main()
