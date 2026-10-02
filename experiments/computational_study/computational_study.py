# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Model-complexity study: DiCEx runtime overhead for models of increasing prediction cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
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

configure_logging(script_name="computational_study", log_dir=LOGS_DIR)

# Seed of the data subsample and of the optimizer. The Random Forests predict with
# n_jobs=-1 (realistic multithreaded timing), whose summation order is nondeterministic,
# so an evaluation count can occasionally differ between runs (see README, Section 7).
SEED = 42
logger = get_logger("computational_study")


def angular_schedule_to_kappas(schedule_deg: list[float], d: int) -> list[float]:
    """Convert angular exploration schedule in degrees to vMF kappas for dimension d.

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


class TimingRegressor:
    """Model adapter that accumulates the prediction time of a black-box model.

    Args:
        model (Any): Base estimator supporting predict().
    """

    def __init__(self, model: Any) -> None:
        """Initialize the timing adapter.

        Args:
            model (Any): Base regression model.
        """
        self.model = model
        self.total_predict_time = 0.0

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict targets while accumulating the elapsed time.

        Args:
            x (NDArray[np.float64]): Input feature matrix.

        Returns:
            NDArray[np.float64]: Model prediction outputs.
        """
        t0 = time.perf_counter()
        out = self.model.predict(x)
        t1 = time.perf_counter()
        self.total_predict_time += t1 - t0
        return out


def main() -> None:
    """Run the model-complexity benchmark across model architectures (CLI entry point)."""
    parser = argparse.ArgumentParser(description="Run computational complexity study.")
    parser.parse_args()

    logger.info("Starting model complexity study")

    optimal_config = load_optimal_config()
    opt_params = optimal_config["optimizer"]
    angular_schedule = opt_params["angular_schedules_deg"]

    mu = 0.35
    sigma = 0.20
    env_std = 0.08
    alpha = 0.10

    out_dir = REPO_ROOT / "experiments" / "computational_study" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_complexity.json"

    logger.info("Loading California Housing...")
    dataset = fetch_california_housing()
    x_train = np.asarray(dataset.data, dtype=float)
    y_train = np.asarray(dataset.target, dtype=float)

    rng = np.random.default_rng(SEED)
    train_indices = rng.choice(len(x_train), size=2000, replace=False)
    x_train_sub = x_train[train_indices]
    y_train_sub = y_train[train_indices]

    d = x_train.shape[1]

    model_variants = [
        {
            "family": "RandomForest",
            "param": "n_estimators",
            "val": 10,
            "model": RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=42),
        },
        {
            "family": "RandomForest",
            "param": "n_estimators",
            "val": 50,
            "model": RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
        },
        {
            "family": "RandomForest",
            "param": "n_estimators",
            "val": 100,
            "model": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        },
        {
            "family": "RandomForest",
            "param": "n_estimators",
            "val": 500,
            "model": RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
        },
        {
            "family": "RandomForest",
            "param": "n_estimators",
            "val": 1000,
            "model": RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=42),
        },
        {
            "family": "MLP",
            "param": "hidden_layers",
            "val": "10",
            "model": MLPRegressor(hidden_layer_sizes=(10,), max_iter=200, random_state=42),
        },
        {
            "family": "MLP",
            "param": "hidden_layers",
            "val": "50",
            "model": MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, random_state=42),
        },
        {
            "family": "MLP",
            "param": "hidden_layers",
            "val": "100",
            "model": MLPRegressor(hidden_layer_sizes=(100,), max_iter=200, random_state=42),
        },
        {
            "family": "MLP",
            "param": "hidden_layers",
            "val": "100_50",
            "model": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42),
        },
        {
            "family": "MLP",
            "param": "hidden_layers",
            "val": "100_100_50",
            "model": MLPRegressor(hidden_layer_sizes=(100, 100, 50), max_iter=200, random_state=42),
        },
    ]

    test_points = x_train_sub[:5]
    results: list[dict[str, object]] = []

    for variant in model_variants:
        family = str(variant["family"])
        param_val = variant["val"]
        model = variant["model"]

        logger.info("Evaluating %s with %s=%s", family, variant["param"], param_val)
        t0_fit = time.perf_counter()
        model.fit(x_train_sub, y_train_sub)
        t1_fit = time.perf_counter()
        logger.info("  Fit time: %.2fs", t1_fit - t0_fit)

        timing_model = TimingRegressor(model)

        kappas = angular_schedule_to_kappas(angular_schedule, d)
        env_cov = np.full(d, env_std**2, dtype=float)
        perturbation = GaussianPerturbation(mu=mu, sigma=sigma, env_cov=env_cov)

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
            "seed": SEED,
        }

        explainer = Dicex(
            model=timing_model,
            task="regression",
            target_class=None,
            perturbation=perturbation,
            alpha=alpha,
            scaler=None,
            **dicex_kwargs,
        )

        times: list[float] = []
        evals: list[float] = []
        for _i, x0 in enumerate(test_points):
            t0 = time.perf_counter()
            res = explainer.explain(x0)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            evals.append(float(res.metadata.get("n_model_evals", 0)))

        mean_time = float(np.mean(times))
        mean_evals = float(np.mean(evals))
        mean_predict_time = timing_model.total_predict_time / len(test_points)

        record = {
            "model_family": family,
            "complexity_param": variant["param"],
            "complexity_value": str(param_val),
            "predict_time_sec": mean_predict_time,
            "n_samples": len(test_points),
            "times_seconds": times,
            "mean_time": mean_time,
            "std_time": float(np.std(times)),
            "mean_evals": mean_evals,
        }
        logger.info("  DiCEx Time: %.2fs ± %.2fs", record["mean_time"], record["std_time"])
        logger.info("  Predict Time (avg): %.3fs", mean_predict_time)
        logger.info(
            "  Model Evals: %.1f, Avg Time per Eval in batch: %.3f µs",
            mean_evals,
            (mean_predict_time / mean_evals) * 1e6 if mean_evals > 0 else 0.0,
        )
        results.append(record)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    logger.info("Model complexity study complete.")


if __name__ == "__main__":
    main()
