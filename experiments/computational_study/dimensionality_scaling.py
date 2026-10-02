# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Dimensionality scaling study: DiCEx runtime on synthetic data of increasing dimension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import yaml

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

from dicex.distributions.gaussian import GaussianPerturbation
from dicex.explainer import Dicex
from dicex.optim.vmf import kappa_from_angular_radius
from experiments.shared.log_config import configure_logging, get_logger

configure_logging(
    script_name="dimensionality_scaling",
    log_dir=LOGS_DIR,
)
logger = get_logger("computational_study.dimensionality_scaling")

DIMS = [4, 8, 16, 32, 64, 128, 256, 512]
SEEDS = [42, 123, 456, 789, 101112]
N_TRAIN = 2000
N_TEST_POINTS = 10

MU = 0.35
SIGMA = 0.20
ENV_STD = 0.08
ALPHA = 0.10


def angular_schedule_to_kappas(schedule_deg: list[float], d: int) -> list[float]:
    """Convert an angular exploration schedule in degrees to vMF concentration parameters kappa.

    Args:
        schedule_deg (list[float]): List of exploration angular radii in degrees.
        d (int): Feature space dimension.

    Returns:
        list[float]: Corresponding vMF concentration parameters kappa.
    """
    kappas: list[float] = []
    for angle_deg in schedule_deg:
        kappa = kappa_from_angular_radius(np.deg2rad(angle_deg), d)
        kappas.append(max(float(kappa), 1e-6))
    return kappas


def load_optimal_config() -> dict[str, Any]:
    """Load the low_recommended hyperparameter configuration preset.

    Returns:
        dict[str, Any]: Parsed hyperparameter dictionary.
    """
    config_path = (
        REPO_ROOT / "experiments" / "hyperparameter_tuning" / "results" / "selected_configs" / "low_recommended.yaml"
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_synthetic_data(
    d: int,
    n_samples: int,
    *,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a synthetic linear dataset with Gaussian features and target noise.

    Args:
        d (int): Number of features.
        n_samples (int): Number of samples.
        seed (int): Random seed.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple (x, y) as float64 arrays.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n_samples, d)).astype(np.float64)
    beta = np.ones(d, dtype=np.float64) / np.sqrt(d)
    y = x @ beta + 0.01 * rng.normal(0, 1, size=n_samples).astype(np.float64)
    return x, y


class TimingRegressor:
    """Model adapter that accumulates the time spent in ``predict()``.

    Args:
        model (object): Base estimator model with predict method.
    """

    def __init__(self, model: object) -> None:
        """Initialize the timing adapter.

        Args:
            model (object): Base estimator model.
        """
        self.model = model
        self.total_predict_time = 0.0

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict targets while measuring elapsed inference time.

        Args:
            x (NDArray[np.float64]): Input feature matrix.

        Returns:
            NDArray[np.float64]: Predictions from base estimator.
        """
        t0 = time.perf_counter()
        out = self.model.predict(x)  # type: ignore[reportAttributeAccessIssue]
        t1 = time.perf_counter()
        self.total_predict_time += t1 - t0
        return out


def main() -> None:
    """Run the dimensionality scaling benchmark (CLI entry point)."""
    parser = argparse.ArgumentParser(description="DiCEx dimensionality scaling with constant inference cost.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run and exit.",
    )
    parser.add_argument(
        "--model",
        choices=["rf", "linear"],
        default="rf",
        help="Model type: 'rf' (RandomForest, O(1) inference wrt d) or 'linear' (Ridge, O(d) inference).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute every configuration, discarding the results already saved (default: resume).",
    )
    args = parser.parse_args()

    logger.info("Starting dimensionality scaling benchmark (model=%s)", args.model)

    optimal_config = load_optimal_config()
    opt_params = optimal_config["optimizer"]
    angular_schedule = opt_params["angular_schedules_deg"]

    out_dir = REPO_ROOT / "experiments" / "computational_study" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dimensionality_{args.model}.json"

    all_results: list[dict[str, object]] = []
    if out_path.exists() and not args.force:
        try:
            with open(out_path) as f:
                all_results = json.load(f)
            logger.info("Loaded %d existing results from %s", len(all_results), out_path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load existing results: %s", e)

    for d in DIMS:
        for seed in SEEDS:
            if any(r.get("d") == d and r.get("seed") == seed for r in all_results):
                logger.info("Configuration d=%d, seed=%d already exists. Skipping.", d, seed)
                continue

            logger.info("--- d=%3d  seed=%d  generating %d samples ---", d, seed, N_TRAIN)

            x, y = generate_synthetic_data(d, N_TRAIN, seed=seed)

            if args.model == "rf":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    n_jobs=-1,
                    random_state=seed,
                )
            else:
                model = Ridge(alpha=1.0, random_state=seed)
            t0_fit = time.perf_counter()
            model.fit(x, y)
            t1_fit = time.perf_counter()
            logger.info("  Fit time: %.2fs  model=%s", t1_fit - t0_fit, args.model)

            timing_model = TimingRegressor(model)

            kappas = angular_schedule_to_kappas(angular_schedule, d)
            env_cov = np.full(d, ENV_STD**2, dtype=float)
            perturbation = GaussianPerturbation(
                mu=MU,
                sigma=SIGMA,
                env_cov=env_cov,
            )

            dicex_kwargs = {
                "kappas": kappas,
                "b_samples": opt_params["b_samples"],
                "n_exploration": opt_params["n_exploration"],
                "n_test": opt_params["n_test"],
                "n_evaluation": opt_params["n_evaluation"],
                "n_bootstrap": opt_params["n_bootstrap"],
                "eta": opt_params["eta"],
                "grid_t": np.linspace(0.0, 1.0, opt_params["grid_t_len"]),
                "max_evals": None,
                "trace_level": "none",
                "seed": seed,
            }

            explainer = Dicex(
                model=timing_model,
                task="regression",
                target_class=None,
                perturbation=perturbation,
                alpha=ALPHA,
                scaler=None,
                **dicex_kwargs,
            )

            rng = np.random.default_rng(seed + 999)
            test_indices = rng.choice(len(x), size=N_TEST_POINTS, replace=False)
            test_points = x[test_indices]

            per_point: list[dict[str, object]] = []
            for i, x0 in enumerate(test_points):
                if args.dry_run:
                    logger.info("  [DRY-RUN] Would explain x0[%d] (d=%d, seed=%d)", i, d, seed)
                    continue

                t0 = time.perf_counter()
                res = explainer.explain(x0)
                t1 = time.perf_counter()

                elapsed = t1 - t0
                meta = res.metadata
                per_point.append(
                    {
                        "d": d,
                        "seed": seed,
                        "test_point_idx": i,
                        "elapsed_seconds": elapsed,
                        "n_model_evals": meta.get("n_model_evals", 0),
                        "n_iterations": meta.get("n_iterations", 0),
                        "n_restarts_performed": meta.get("n_restarts_performed", 0),
                        "n_accepted_moves": meta.get("n_accepted_moves", 0),
                        "robust_value": float(res.robust_value),
                    }
                )

            if args.dry_run:
                continue

            mean_predict_time = timing_model.total_predict_time / N_TEST_POINTS

            elapsed_vals = [float(p["elapsed_seconds"]) for p in per_point]
            evals_vals = [float(p["n_model_evals"]) for p in per_point]
            record: dict[str, object] = {
                "d": d,
                "seed": seed,
                "n_test_points": N_TEST_POINTS,
                "mean_elapsed_seconds": float(np.mean(elapsed_vals)),
                "std_elapsed_seconds": float(np.std(elapsed_vals)),
                "mean_predict_time_sec": mean_predict_time,
                "mean_n_model_evals": float(np.mean(evals_vals)),
                "std_n_model_evals": float(np.std(evals_vals)),
                "mean_n_iterations": float(np.mean([float(p["n_iterations"]) for p in per_point])),
                "mean_n_restarts": float(np.mean([float(p["n_restarts_performed"]) for p in per_point])),
                "points": per_point,
            }

            logger.info(
                "  d=%3d seed=%d  elapsed=%.3f±%.3fs  predict=%.4fs  evals=%.0f±%.0f  iters=%.1f  restarts=%.1f",
                d,
                seed,
                float(record["mean_elapsed_seconds"]),
                float(record["std_elapsed_seconds"]),
                mean_predict_time,
                float(record["mean_n_model_evals"]),
                float(record["std_n_model_evals"]),
                float(record["mean_n_iterations"]),
                float(record["mean_n_restarts"]),
            )

            all_results.append(record)

            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

    if args.dry_run:
        logger.info("Dry-run complete. No results saved.")
        return

    logger.info("Dimensionality scaling complete — %d configurations saved to %s", len(all_results), out_path)
    logger.info("%-4s  %12s  %12s  %8s  %6s", "d", "elapsed(s)", "predict(s)", "evals", "iters")
    for d in DIMS:
        sub = [r for r in all_results if r.get("d") == d]
        if not sub:
            continue
        mean_elapsed = float(np.mean([float(r["mean_elapsed_seconds"]) for r in sub]))
        mean_predict = float(np.mean([float(r["mean_predict_time_sec"]) for r in sub]))
        mean_evals = float(np.mean([float(r["mean_n_model_evals"]) for r in sub]))
        mean_iters = float(np.mean([float(r["mean_n_iterations"]) for r in sub]))
        logger.info("%4d  %10.3fs  %10.4fs  %8.0f  %6.1f", d, mean_elapsed, mean_predict, mean_evals, mean_iters)


if __name__ == "__main__":
    main()
