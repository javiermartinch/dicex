# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Comparative audit of DiCEx against LIME, Wachter CF, nominal, and random-direction baselines."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import UTC, datetime
import importlib.util
import logging
from multiprocessing import Pool
from pathlib import Path
import sys
import time
import traceback
from typing import TYPE_CHECKING, Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml

from dicex.distributions.gaussian import GaussianPerturbation
from dicex.explainer import Dicex
from dicex.preprocessing.scaling import DicexScaler
from experiments.shared.baselines import (
    run_nominal_baseline,
    run_random_direction_baseline,
)
from experiments.shared.helpers import angular_schedule_to_kappas, get_predict_fn
from experiments.shared.loader import load_trained_artifacts
from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.metrics import evaluate_metrics
from experiments.shared.persistence import ensure_layout, experiment_layout
from experiments.shared.query_points import compute_query_points_on_test
from experiments.shared.specs import NoiseRiskConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from experiments.shared.query_points import QueryPoint

logger = get_logger("comparative_audit")


class NoiseScenario(TypedDict):
    """Named noise scenario of the audit grid."""

    mu: float
    sigma: float
    env_std: float
    label: str


METHODS = ("dicex", "lime", "cf_wachter", "nominal", "random_direction")
QUERY_PERCENTILES = (0.10, 0.50, 0.90)
ALPHAS = [0.10, 0.25, 0.50, 0.75, 0.90]
NOISE_SCENARIOS: list[NoiseScenario] = [
    {"mu": 0.25, "sigma": 0.25, "env_std": 0.05, "label": "balanced"},
    {"mu": 0.5, "sigma": 0.25, "env_std": 0.05, "label": "dominant"},
    {"mu": 0.25, "sigma": 0.05, "env_std": 0.01, "label": "low_noise"},
    {"mu": 0.25, "sigma": 0.35, "env_std": 0.07, "label": "elevated_noise"},
]


def run_lime_baseline(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    sigma_perturb: float = 0.1,
    n_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Estimate a local linear surrogate direction via LIME in scaled space.

    Args:
        predict_fn (Callable[[np.ndarray], np.ndarray]): Model prediction function.
        x0 (np.ndarray): Scaled factual query point.
        sigma_perturb (float): Perturbation scale for local sampling.
        n_samples (int): Number of perturbed samples for surrogate fitting.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Normalized linear surrogate direction vector.
    """
    rng = np.random.default_rng(seed)
    d = len(x0)

    perturbations = rng.normal(scale=sigma_perturb, size=(n_samples, d))
    samples = x0 + perturbations

    preds = predict_fn(samples)
    pred_x0 = predict_fn(x0[None, :])[0]

    kernel_width = 0.75 * np.sqrt(d)
    distances = np.linalg.norm(perturbations, axis=1)
    weights = np.exp(-(distances**2) / (kernel_width**2))

    w_half = np.sqrt(weights)
    x_weighted = perturbations * w_half[:, None]
    y_weighted = (preds - pred_x0) * w_half

    alpha = 1e-4
    xtx = x_weighted.T @ x_weighted + alpha * np.eye(d)
    xty = x_weighted.T @ y_weighted
    try:
        w = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        w = xty

    norm_w = np.linalg.norm(w)
    if norm_w < 1e-8:
        return np.zeros_like(x0)
    return w / norm_w


def determine_wachter_target(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    x_train_scaled: np.ndarray,
    task: str,
) -> float:
    """Determine a plausible target prediction threshold for Wachter counterfactuals.

    Args:
        predict_fn (Callable[[np.ndarray], np.ndarray]): Model prediction function.
        x0 (np.ndarray): Scaled factual query point.
        x_train_scaled (np.ndarray): Scaled training features.
        task (str): Task identifier ('classification' or 'regression').

    Returns:
        float: Target prediction value for counterfactual search.
    """
    y0 = float(predict_fn(x0[None, :])[0])
    if task == "classification":
        return float(np.clip(y0 + 0.1, 0.01, 0.99))
    train_preds = predict_fn(x_train_scaled)
    std_y = float(np.std(train_preds))
    if std_y < 1e-5:
        std_y = 1.0
    return float(y0 + 0.2 * std_y)


def run_wachter_baseline(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    target_value: float,
    x_train_scaled: np.ndarray,
    _seed: int = 42,
) -> np.ndarray:
    """Compute counterfactual direction via Wachter distance minimization.

    Args:
        predict_fn (Callable[[np.ndarray], np.ndarray]): Model prediction function.
        x0 (np.ndarray): Scaled factual query point.
        target_value (float): Target prediction threshold.
        x_train_scaled (np.ndarray): Scaled training data for initialization.
        _seed (int): Random seed.

    Returns:
        np.ndarray: Normalized directional vector pointing to counterfactual.
    """
    preds = predict_fn(x_train_scaled)
    valid_indices = np.where(preds >= target_value)[0]

    if len(valid_indices) == 0:
        best_idx = np.argmax(preds)
        cf_init = x_train_scaled[best_idx]
        target_value = float(preds[best_idx])
    else:
        dists = np.linalg.norm(x_train_scaled[valid_indices] - x0, axis=1)
        best_idx = valid_indices[np.argmin(dists)]
        cf_init = x_train_scaled[best_idx]

    def wachter_objective(x: np.ndarray) -> float:
        dist_sq = np.sum((x - x0) ** 2)
        pred = predict_fn(x[None, :])[0]
        penalty = 1e6 * (target_value - pred) ** 2 if pred < target_value else 0.0
        return float(dist_sq + penalty)

    res = minimize(wachter_objective, cf_init, method="Nelder-Mead", options={"maxiter": 300})

    direction = res.x - x0
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.zeros_like(x0)
    return direction / norm


def _load_visualize_main() -> Any:
    """Load the entry point of the comparative audit visualization script.

    Returns:
        Any: The main function of visualize_and_report.

    Raises:
        RuntimeError: If visualize_and_report cannot be loaded.
    """
    module_path = Path(__file__).with_name("visualize_and_report.py")
    module_name = "comparative_audit_visualize"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load visualize module from {module_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.main


def _dicex_optimizer_kwargs(*, opt: dict[str, Any], exec_cfg: dict[str, Any], d: int, seed: int) -> dict[str, Any]:
    """Build optimizer keyword arguments for DiCEx explainer.

    Args:
        opt (dict[str, Any]): Optimizer hyperparameter dictionary.
        exec_cfg (dict[str, Any]): Execution configuration dictionary.
        d (int): Feature dimensionality.
        seed (int): Random seed.

    Returns:
        dict[str, Any]: Formatted keyword arguments for DiCEx explainer.
    """
    return {
        "kappas": angular_schedule_to_kappas(opt["angular_schedules_deg"], d),
        "b_samples": opt["b_samples"],
        "n_exploration": opt["n_exploration"],
        "n_test": opt["n_test"],
        "n_evaluation": opt["n_evaluation"],
        "n_bootstrap": opt["n_bootstrap"],
        "eta": opt["eta"],
        "grid_t": np.linspace(0.0, 1.0, opt["grid_t_len"]),
        "max_evals": None,
        "trace_level": "none",
        "min_runtime_seconds": None,
        "timeout_seconds": exec_cfg.get("timeout_seconds"),
        "seed": seed,
    }


def run_single_comparison(
    *,
    artifact: Any,
    query_point: QueryPoint,
    noise_config: NoiseRiskConfig,
    noise_label: str,
    seed: int,
    _analytic_solutions: dict,
    precomputed_static: dict[str, np.ndarray],
    c_nominal: np.ndarray,
    pred_range_test: float,
) -> list[dict[str, Any]]:
    """Evaluate all baseline methods on a single artifact, query point, and noise level.

    Args:
        artifact (Any): Trained model artifact container.
        query_point (QueryPoint): Target factual query point.
        noise_config (NoiseRiskConfig): Perturbation and risk configuration.
        noise_label (str): Label identifier for the noise scenario.
        seed (int): Random seed for reproducibility.
        _analytic_solutions (dict): Dictionary of precomputed analytic solutions.
        precomputed_static (dict[str, np.ndarray]): Precomputed static baseline directions.
        c_nominal (np.ndarray): Precomputed nominal baseline direction.
        pred_range_test (float): Prediction range on test split for normalization.

    Returns:
        list[dict[str, Any]]: Evaluation records for all audited methods.
    """
    x0 = np.asarray(query_point.x0, dtype=float)
    d = len(x0)
    predict_fn = get_predict_fn(artifact)

    # Load the optimizer preset matching this artifact's dimension bucket
    config_path = (
        REPO_ROOT
        / "experiments"
        / "hyperparameter_tuning"
        / "results"
        / "selected_configs"
        / f"{artifact.dimension_bucket}_mid.yaml"
    )
    with open(config_path) as f:
        payload = yaml.safe_load(f)
    opt = payload["optimizer"]

    fresh_samples = opt["n_evaluation"]
    random_internal_samples = max(32, opt["n_test"])

    # Reconstruct the scaler so that every method operates in the scaled space
    scaler = DicexScaler()
    if artifact.scaler is not None:
        scaler.mean_ = np.asarray(artifact.scaler.mean_, dtype=float)
        scaler.std_ = np.asarray(artifact.scaler.scale_, dtype=float)
    else:
        scaler.fit(artifact.x_train)

    x0_scaled = scaler.transform(x0.reshape(1, -1)).reshape(-1)

    # Prediction function on scaled inputs
    def predict_fn_scaled(x_scaled: np.ndarray) -> np.ndarray:
        if x_scaled.ndim == 1:
            x_raw = scaler.inverse_transform(x_scaled[None, :])
            return np.asarray(predict_fn(x_raw).flatten()[0], dtype=float)
        x_raw = scaler.inverse_transform(x_scaled)
        return np.asarray(predict_fn(x_raw), dtype=float)

    # Run DiCEx in scaled space to match the evaluated noise model.
    class ScaledMockModel:
        def predict(self, x):
            return predict_fn_scaled(x)

        def predict_proba(self, x):
            p = predict_fn_scaled(x)
            return np.column_stack([1 - p, p])

    perturbation = GaussianPerturbation(
        mu=noise_config.mu, sigma=noise_config.sigma, env_cov=np.full(d, noise_config.env_std**2, dtype=float)
    )

    dicex_t0 = time.perf_counter()
    explainer = Dicex(
        model=ScaledMockModel(),
        task=artifact.task,
        target_class=1 if artifact.task == "classification" else None,
        perturbation=perturbation,
        alpha=noise_config.alpha,
        scaler=None,
        verbose="none",
        **_dicex_optimizer_kwargs(opt=opt, exec_cfg=payload.get("execution", {}), d=d, seed=seed),
    )
    dicex_res_obj = explainer.explain(x0_scaled)
    dicex_elapsed = time.perf_counter() - dicex_t0

    c_dicex_scaled = dicex_res_obj.direction
    baseline_cvar = dicex_res_obj.metadata.get("baseline_cvar", -np.inf)
    if dicex_res_obj.robust_value <= baseline_cvar:
        logger.info("DiCEx optimization indicates inaction is preferred. Setting direction to zero.")
        c_dicex_scaled = np.zeros_like(x0_scaled)

    n_model_evals = dicex_res_obj.metadata.get("n_model_evals", 0)

    directions_scaled = {
        "dicex": c_dicex_scaled,
        "lime": precomputed_static["lime"],
        "cf_wachter": precomputed_static["cf_wachter"],
        "nominal": c_nominal,
        "random_direction": run_random_direction_baseline(
            predict_fn_scaled,
            x0_scaled,
            noise_config.mu,
            noise_config.sigma,
            noise_config.env_std,
            noise_config.alpha,
            n_internal_samples=random_internal_samples,
            k_directions=16,
            seed=seed + 17,
        ),
    }

    rows: list[dict[str, Any]] = []
    for method_index, method_name in enumerate(METHODS):
        metrics = evaluate_metrics(
            predict_fn_scaled,
            x0_scaled,
            np.asarray(directions_scaled[method_name], dtype=float),
            noise_config.mu,
            noise_config.sigma,
            noise_config.env_std,
            noise_config.alpha,
            n_fresh_samples=fresh_samples,
            seed=seed + 1000 + method_index,
        )
        rows.append(
            {
                "artifact_id": artifact.artifact_id,
                "dataset_name": artifact.dataset_name,
                "model_family": artifact.model_family,
                "task": artifact.task,
                "dimension_bucket": artifact.dimension_bucket,
                "n_features": d,
                "query_id": query_point.query_id,
                "query_percentile": query_point.percentile,
                "noise_label": noise_label,
                "mu": noise_config.mu,
                "sigma": noise_config.sigma,
                "env_std": noise_config.env_std,
                "alpha": noise_config.alpha,
                "method": method_name,
                "robust_gain": metrics["robust_gain"],
                "cvar_norm_maxmin": metrics["robust_gain"] / pred_range_test,
                "mean_gain": metrics["mean_gain"],
                "mean_gain_norm_maxmin": metrics["mean_gain"] / pred_range_test,
                "failure_rate": metrics["failure_rate"],
                "volatility": metrics["volatility"],
                "volatility_norm_maxmin": metrics["volatility"] / pred_range_test,
                "nominal_gain": metrics["nominal_gain"],
                "nominal_gain_norm_maxmin": metrics["nominal_gain"] / pred_range_test,
                "action_magnitude": metrics["action_magnitude"],
                "elapsed_seconds": dicex_elapsed if method_name == "dicex" else 0.0,
                "n_model_evals": n_model_evals if method_name == "dicex" else 0,
                "b_samples": opt["b_samples"],
                "n_exploration": opt["n_exploration"],
                "n_test": opt["n_test"],
                "n_evaluation": opt["n_evaluation"],
                "n_bootstrap": opt["n_bootstrap"],
                "eta": opt["eta"],
                "angular_schedule_deg": str(list(opt["angular_schedules_deg"])),
                "grid_t_len": opt["grid_t_len"],
            }
        )
    return rows


def _run_task(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Run a single benchmark comparison task in a worker process.

    Args:
        task_dict (dict[str, Any]): Task configuration dictionary.

    Returns:
        dict[str, Any]: Result payload with execution status and metrics.
    """
    try:
        t = task_dict
        res = run_single_comparison(
            artifact=t["artifact"],
            query_point=t["query_point"],
            noise_config=t["noise_config"],
            noise_label=t["noise_label"],
            seed=t["seed"],
            _analytic_solutions=t["analytic_solutions"],
            precomputed_static=t["precomputed_static"],
            c_nominal=t["c_nominal"],
            pred_range_test=t["pred_range_test"],
        )
    except Exception as e:  # noqa: BLE001
        logger.error(traceback.format_exc())
        return {"status": "error", "error": f"{e!s}\n{traceback.format_exc()}", "t": task_dict}
    else:
        return {"status": "success", "res": res, "t": t}


def _precompute_static_single(task: tuple[Any, QueryPoint, int, int]) -> tuple[tuple[str, str], dict[str, np.ndarray]]:
    """Precompute static baseline directions (LIME and Wachter) for a single query point.

    Args:
        task (tuple[Any, QueryPoint, int, int]): Tuple containing artifact, query point, seed, and index.

    Returns:
        tuple[tuple[str, str], dict[str, np.ndarray]]: Artifact/query key and dictionary of computed directions.
    """
    artifact, query_point, seed, counter = task
    predict_fn = get_predict_fn(artifact)

    scaler = DicexScaler()
    if artifact.scaler is not None:
        scaler.mean_ = np.asarray(artifact.scaler.mean_, dtype=float)
        scaler.std_ = np.asarray(artifact.scaler.scale_, dtype=float)
    else:
        scaler.fit(artifact.x_train)

    x_train_scaled = scaler.transform(artifact.x_train)
    x0 = np.asarray(query_point.x0, dtype=float)
    x0_scaled = scaler.transform(x0[None, :])[0]

    def predict_fn_scaled(x_scaled: np.ndarray) -> np.ndarray:
        if x_scaled.ndim == 1:
            x_raw = scaler.inverse_transform(x_scaled[None, :])
            return np.asarray(predict_fn(x_raw).flatten()[0], dtype=float)
        x_raw = scaler.inverse_transform(x_scaled)
        return np.asarray(predict_fn(x_raw), dtype=float)

    query_seed = seed + counter
    c_lime = run_lime_baseline(predict_fn_scaled, x0_scaled, seed=query_seed)
    target_val = determine_wachter_target(predict_fn_scaled, x0_scaled, x_train_scaled, artifact.task)
    c_wachter = run_wachter_baseline(predict_fn_scaled, x0_scaled, target_val, x_train_scaled, _seed=query_seed)

    return (artifact.artifact_id, query_point.query_id), {"lime": c_lime, "cf_wachter": c_wachter}


def precompute_static_directions(
    artifacts: list[Any], seed: int, workers: int = 1
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Precompute LIME and Wachter baseline directions in parallel across query points.

    Args:
        artifacts (list[Any]): Collection of trained model artifacts.
        seed (int): Base random seed.
        workers (int): Number of parallel worker processes.

    Returns:
        dict[tuple[str, str], dict[str, np.ndarray]]: Map of (artifact_id, query_id) to directions.
    """
    # Each query point uses seed + its (1-based) index within the artifact.
    tasks = [
        (artifact, query_point, seed, counter + 1)
        for artifact in artifacts
        for counter, query_point in enumerate(artifact.query_points)
    ]

    logger.info("Precomputing %d static directions (LIME & Wachter) with %d workers...", len(tasks), workers)

    static_dirs: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    completed = 0
    with Pool(workers) as pool:
        for key, res in pool.imap_unordered(_precompute_static_single, tasks):
            completed += 1
            static_dirs[key] = res
            if completed % 20 == 0 or completed == len(tasks):
                logger.info("  Static precomputation progress: %d/%d", completed, len(tasks))

    return static_dirs


def _precompute_nominal_single(
    task: tuple[Any, QueryPoint, float, int, int],
) -> tuple[tuple[str, str, float], np.ndarray]:
    """Precompute nominal baseline direction for a single query point and action magnitude.

    Args:
        task (tuple[Any, QueryPoint, float, int, int]): Tuple of artifact, query point, mu, seed, and index.

    Returns:
        tuple[tuple[str, str, float], np.ndarray]: Key tuple and computed nominal direction vector.
    """
    artifact, query_point, mu, seed, _counter = task
    predict_fn = get_predict_fn(artifact)

    scaler = DicexScaler()
    if artifact.scaler is not None:
        scaler.mean_ = np.asarray(artifact.scaler.mean_, dtype=float)
        scaler.std_ = np.asarray(artifact.scaler.scale_, dtype=float)
    else:
        scaler.fit(artifact.x_train)

    x0 = np.asarray(query_point.x0, dtype=float)
    x0_scaled = scaler.transform(x0[None, :])[0]

    def predict_fn_scaled(x_scaled: np.ndarray) -> np.ndarray:
        if x_scaled.ndim == 1:
            x_raw = scaler.inverse_transform(x_scaled[None, :])
            return np.asarray(predict_fn(x_raw).flatten()[0], dtype=float)
        x_raw = scaler.inverse_transform(x_scaled)
        return np.asarray(predict_fn(x_raw), dtype=float)

    nominal_dir_scaled = run_nominal_baseline(predict_fn_scaled, x0_scaled, mu, seed=seed)
    # All directions are evaluated in the scaled feature space (see run_single_comparison).
    return ((artifact.artifact_id, query_point.query_id, mu), nominal_dir_scaled)


def precompute_nominal_directions(
    artifacts: list[Any],
    mus: list[float],
    seed: int,
    workers: int = 1,
) -> dict[tuple[str, str, float], np.ndarray]:
    """Precompute nominal baseline directions in parallel across query points and action bounds.

    Args:
        artifacts (list[Any]): Collection of trained model artifacts.
        mus (list[float]): List of action magnitude constraints.
        seed (int): Base random seed.
        workers (int): Number of parallel worker processes.

    Returns:
        dict[tuple[str, str, float], np.ndarray]: Map of (artifact_id, query_id, mu) to nominal directions.
    """
    tasks = []
    counter = 0
    for artifact in artifacts:
        for query_point in artifact.query_points:
            for mu in mus:
                counter += 1
                tasks.append((artifact, query_point, mu, seed + counter, counter))

    logger.info("Precomputing %d nominal directions with %d workers...", len(tasks), workers)

    nominal_dirs: dict[tuple[str, str, float], np.ndarray] = {}
    completed = 0
    with Pool(workers) as pool:
        for key, nominal_dir in pool.imap_unordered(_precompute_nominal_single, tasks):
            completed += 1
            nominal_dirs[key] = nominal_dir
            if completed % 20 == 0 or completed == len(tasks):
                logger.info("  Nominal precomputation progress: %d/%d", completed, len(tasks))

    return nominal_dirs


def main() -> None:
    """Run the comparative baseline audit across the registered datasets and models (CLI entry point)."""
    parser = ArgumentParser(description="Run the comparative baseline audit benchmark.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers for parallel execution (default: 1).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="all",
        choices=["low", "medium", "high", "all"],
        help="Dimension bucket to run.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["linear", "logistic", "random_forest", "xgboost", "mlp"],
        help="Model families to include.",
    )
    parser.add_argument("--skip-visualize", action="store_true", help="Skip visualization step.")
    args = parser.parse_args()

    seed = 42

    layout = experiment_layout(results_root=REPO_ROOT / "experiments" / "comparative_audit" / "results")
    ensure_layout(layout)

    results_dir = REPO_ROOT / "experiments" / "comparative_audit" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    configure_logging(layout, "comparative_audit", log_dir=log_dir)
    logging.getLogger("dicex").setLevel(logging.WARNING)

    bucket_arg = None if args.bucket == "all" else args.bucket
    artifacts: list[Any] = load_trained_artifacts(layout, bucket=bucket_arg, model_families=args.models)
    artifacts = [a for a in artifacts if a.model_family in args.models]

    logger.info("Computing query points at %s...", QUERY_PERCENTILES)
    compute_query_points_on_test(artifacts, QUERY_PERCENTILES)

    analytic_solutions = {}

    # Precompute static and nominal directions (in parallel)
    static_dirs = precompute_static_directions(artifacts, seed, workers=args.workers)
    unique_mus = list({scenario["mu"] for scenario in NOISE_SCENARIOS})
    nominal_dirs = precompute_nominal_directions(artifacts, unique_mus, seed, workers=args.workers)

    total_tasks = sum(len(a.query_points) for a in artifacts) * len(NOISE_SCENARIOS) * len(ALPHAS)
    logger.info("Prepared %d artifact/noise/alpha tasks (%d methods each).", total_tasks, len(METHODS))

    # Precompute pred_range_test for each artifact
    artifact_ranges = {}
    for artifact in artifacts:
        predict_fn = get_predict_fn(artifact)
        test_preds = predict_fn(artifact.x_test)
        rng_val = float(np.max(test_preds) - np.min(test_preds))
        if rng_val < 1e-6:
            rng_val = 1.0
        artifact_ranges[artifact.artifact_id] = rng_val

    tasks = []
    task_counter = 0
    for artifact in artifacts:
        pred_range_test = artifact_ranges[artifact.artifact_id]
        for query_point in artifact.query_points:
            for scenario in NOISE_SCENARIOS:
                for alpha in ALPHAS:
                    task_counter += 1
                    noise_config = NoiseRiskConfig(
                        mu=scenario["mu"],
                        sigma=scenario["sigma"],
                        env_std=scenario["env_std"],
                        alpha=alpha,
                    )
                    tasks.append(
                        {
                            "artifact": artifact,
                            "query_point": query_point,
                            "noise_config": noise_config,
                            "noise_label": scenario["label"],
                            "seed": seed + task_counter,
                            "analytic_solutions": analytic_solutions,
                            "precomputed_static": static_dirs[(artifact.artifact_id, query_point.query_id)],
                            "c_nominal": nominal_dirs[(artifact.artifact_id, query_point.query_id, scenario["mu"])],
                            "pred_range_test": pred_range_test,
                        }
                    )

    logger.info("Executing %d tasks in parallel with %d workers...", len(tasks), args.workers)

    rows: list[dict[str, Any]] = []
    completed_count = 0

    with Pool(args.workers) as pool:
        for output in pool.imap_unordered(_run_task, tasks):
            completed_count += 1
            t = output["t"]
            if output["status"] == "success":
                res = output["res"]
                rows.extend(res)
                remaining = len(tasks) - completed_count
                logger.info(
                    "[%d/%d] Completed %s | %s | %s | alpha=%.2f (%d remaining)",
                    completed_count,
                    len(tasks),
                    t["artifact"].artifact_id,
                    t["query_point"].query_id,
                    t["noise_label"],
                    t["noise_config"].alpha,
                    remaining,
                )
            else:
                remaining = len(tasks) - completed_count
                logger.error(
                    "[%d/%d] Error running %s | %s | %s | alpha=%.2f (%d remaining): %s",
                    completed_count,
                    len(tasks),
                    t["artifact"].artifact_id,
                    t["query_point"].query_id,
                    t["noise_label"],
                    t["noise_config"].alpha,
                    remaining,
                    output["error"],
                )

    df = pd.DataFrame(rows)
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    raw_results_path = results_dir / f"raw_results_{timestamp}.csv"
    df.to_csv(raw_results_path, index=False)
    logger.info("Raw results saved to %s", raw_results_path)

    # The reporting scripts (this folder and computational_study) read paper_results.csv.
    paper_results_path = results_dir / "paper_results.csv"
    df.to_csv(paper_results_path, index=False)
    logger.info("Paper results saved to %s", paper_results_path)

    if not args.skip_visualize:
        logger.info("Generating plots and report summaries...")
        run_visualize = _load_visualize_main()
        run_visualize()


if __name__ == "__main__":
    main()
