# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Compare DiCEx against closed-form optimal directions for linear and logistic models.

Runs the analytic-convergence grid over optimizer presets, noise scenarios, and
risk levels, and writes one raw results CSV per execution.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dicex.analytical.linear import linear_directional_functional
from dicex.analytical.logistic import logistic_directional_functional
from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.core.functional import directional_improvement
from dicex.core.scoring import ModelScorer
from dicex.distributions.gaussian import GaussianPerturbation
from dicex.preprocessing.scaling import DicexScaler
from experiments.shared.analytic_solutions import get_solutions_path, load_analytic_solutions, make_solutions_key
from experiments.shared.evaluation import (
    ScaledModelWrapper,
    run_artifact_explanation,
)
from experiments.shared.helpers import repo_relative

if TYPE_CHECKING:
    from experiments.shared.bundles import TrainedModelArtifact
from experiments.shared.helpers import angular_schedule_to_kappas, get_predict_fn
from experiments.shared.loader import load_trained_artifacts
from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.metrics import angle_deg, minimal_angle_to_projection_set_deg
from experiments.shared.persistence import (
    ensure_layout,
    experiment_layout,
)
from experiments.shared.query_points import compute_query_points_on_test
from experiments.shared.specs import NoiseRiskConfig

logger = get_logger("analytic_convergence")

# Load configuration parameters
CONFIG_PATH = Path(__file__).resolve().parent / "experiment_config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    _FULL_CONFIG = yaml.safe_load(f)

_PRESET_OVERRIDE = os.environ.get("DICEX_OVERRIDE_PRESET")
if _PRESET_OVERRIDE and _PRESET_OVERRIDE in _FULL_CONFIG["optimizer"]["presets"]:
    _ACTIVE_PRESET = _PRESET_OVERRIDE
else:
    _ACTIVE_PRESET = _FULL_CONFIG["optimizer"]["default"]
DICEX_CONFIG = _FULL_CONFIG["optimizer"]["presets"][_ACTIVE_PRESET]
DICEX_PRESETS = _FULL_CONFIG["optimizer"]["presets"]
EXPERIMENT_CONFIG = _FULL_CONFIG["experiment"]


def run_single_comparison(
    artifact: TrainedModelArtifact,
    x0: np.ndarray,
    query_id: str,
    noise_config: NoiseRiskConfig,
    seed: int,
    analytic_solutions: dict[str, Any],
    pred_range_test: float,
    noise_label: str = "",
    preset_name: str = "high",
    dicex_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single comparison between analytical and DiCEx solutions.

    Args:
        artifact (TrainedModelArtifact): Trained model artifact with scaler and metadata.
        x0 (np.ndarray): Baseline query point.
        query_id (str): Identifier for the query point configuration.
        noise_config (NoiseRiskConfig): Perturbation noise parameters (mu, sigma, env_std, alpha).
        seed (int): Random seed for the DiCEx optimizer.
        analytic_solutions (dict[str, Any]): Precomputed analytical solutions keyed by solution key.
        pred_range_test (float): Prediction range on test set for CVaR normalization.
        noise_label (str): Human-readable label for the noise scenario.
        preset_name (str): Name of the optimizer preset being evaluated.
        dicex_config (dict[str, Any] | None): Explicit configuration dictionary for the optimizer preset.

    Returns:
        dict[str, Any]: Dictionary with comparison metrics including value differences, angular
            deviations, and configuration metadata.
    """
    t0 = time.perf_counter()

    env_cov = np.full(len(x0), noise_config.env_std**2, dtype=float)
    perturbation = GaussianPerturbation(mu=noise_config.mu, sigma=noise_config.sigma, env_cov=env_cov)

    sol_key = make_solutions_key(artifact.artifact_id, str(query_id), noise_config)
    if sol_key not in analytic_solutions:
        msg = f"Analytical solution not found in cache for {artifact.artifact_id} with config {sol_key}"
        raise ValueError(msg)

    cached_res = analytic_solutions[sol_key]
    analytic_dir_norm = cached_res["analytic_dir_norm"]
    analytic_val = cached_res["analytic_val"]
    beta_norm = cached_res["beta_norm"]
    intercept_norm = cached_res["intercept_norm"]
    x0_eval = cached_res["x0_eval"]

    cfg = dicex_config if dicex_config is not None else DICEX_CONFIG
    b_samples = cfg["b_samples"]
    n_exploration = cfg["n_exploration"]
    n_test = cfg["n_test"]
    n_evaluation = cfg["n_evaluation"]
    n_bootstrap = cfg["n_bootstrap"]

    d = len(x0)
    kappas = angular_schedule_to_kappas(cfg["angular_schedule_deg"], d)

    optimizer_kwargs = {
        "kappas": kappas,
        "b_samples": b_samples,
        "n_exploration": n_exploration,
        "n_test": n_test,
        "n_evaluation": n_evaluation,
        "n_bootstrap": n_bootstrap,
        "eta": cfg["eta"],
        "grid_t": np.linspace(0.0, 1.0, cfg["grid_t_len"]),
        "max_evals": None,
        "trace_level": "none",
        "min_runtime_seconds": None,
        "timeout_seconds": None,
        "seed": seed,
    }

    dicex_res = run_artifact_explanation(
        artifact,
        x0=x0,
        noise_config=noise_config,
        optimizer_kwargs=optimizer_kwargs,
    )

    meta = dicex_res.get("metadata", {})
    if "best_sphere_direction" in meta:
        dicex_dir_norm = np.asarray(meta["best_sphere_direction"], dtype=float)
    elif "direction_scaled" in meta:
        dicex_dir_norm = np.asarray(meta["direction_scaled"], dtype=float)
    else:
        dicex_dir_norm = np.asarray(dicex_res["direction"], dtype=float)

    norm_val = np.linalg.norm(dicex_dir_norm)
    if norm_val > 1e-12:
        dicex_dir_norm = dicex_dir_norm / norm_val
    else:
        msg = "Direction norm is zero; cannot normalize."
        raise ValueError(msg)

    if "best_sphere_cvar" in meta:
        dicex_val = float(meta["best_sphere_cvar"])
    else:
        dicex_val = float(dicex_res["robust_value"])

    if artifact.model_family == "linear":
        dicex_val_analytic = float(
            linear_directional_functional(beta_norm, perturbation, noise_config.alpha, dicex_dir_norm)
        )
    elif artifact.model_family == "logistic":
        dicex_val_analytic = float(
            logistic_directional_functional(
                beta_norm, intercept_norm, x0_eval, perturbation, noise_config.alpha, dicex_dir_norm
            )
        )

    epsilon_val = EXPERIMENT_CONFIG["epsilon"]
    n_baseline_dirs = EXPERIMENT_CONFIG["n_baseline_random_dirs"]

    val_diff = float(dicex_val - analytic_val)
    abs_val_diff = float(abs(val_diff))
    rel_val_diff = float(abs_val_diff / (abs(analytic_val) + epsilon_val))

    val_diff_direction = float(dicex_val_analytic - analytic_val)
    rel_val_diff_direction = float(abs(val_diff_direction) / (abs(analytic_val) + epsilon_val))

    sampling_abs_val_diff = float(abs(dicex_val - dicex_val_analytic))
    sampling_rel_val_diff = sampling_abs_val_diff / (abs(dicex_val_analytic) + epsilon_val)

    if artifact.scaler is not None:
        scaler = DicexScaler()
        scaler.mean_ = np.asarray(artifact.scaler.mean_, dtype=float)
        scaler.std_ = np.asarray(artifact.scaler.scale_, dtype=float)
    else:
        scaler = None

    if scaler is not None:
        x0_norm = scaler.transform(x0.reshape(1, -1)).reshape(-1)
        perturbation_norm = perturbation
        model_norm = scaler.wrap_model(
            ScaledModelWrapper(artifact.model, artifact.scaler) if artifact.scaler is not None else artifact.model
        )
    else:
        x0_norm = x0.copy()
        perturbation_norm = perturbation
        model_norm = artifact.model

    scorer = ModelScorer(model_norm, target_class=artifact.target_class)
    rng_baseline = np.random.default_rng(seed + 9999)
    final_noises = perturbation_norm.sample(n_evaluation, d=d, rng=rng_baseline)
    final_env_noises = perturbation_norm.sample_env(n_evaluation, d, rng=rng_baseline)

    baseline_cvars = []
    for _ in range(n_baseline_dirs):
        rand_dir = rng_baseline.standard_normal(d)
        rand_dir_norm = np.linalg.norm(rand_dir)
        if rand_dir_norm > epsilon_val:
            rand_dir /= rand_dir_norm
        else:
            rand_dir = np.zeros(d)
            rand_dir[0] = 1.0

        scorer.reset_baseline()
        z_samples = directional_improvement(scorer, x0_norm, rand_dir, final_noises, final_env_noises)
        cvar_val = float(lower_tail_cvar_empirical(z_samples, noise_config.alpha))
        baseline_cvars.append(cvar_val)

    baseline_random_cvar_median = float(np.median(baseline_cvars))

    rel_error_to_baseline = sampling_abs_val_diff / (
        abs(dicex_val_analytic - baseline_random_cvar_median) + epsilon_val
    )
    rel_val_diff_norm = abs_val_diff / (pred_range_test + epsilon_val)
    baseline_improvement_norm = float(abs(analytic_val - baseline_random_cvar_median)) / (pred_range_test + epsilon_val)
    baseline_rel_error = float(abs(analytic_val - baseline_random_cvar_median) / (abs(analytic_val) + epsilon_val))

    angle_diff_sol_norm = angle_deg(analytic_dir_norm, dicex_dir_norm)
    t_star = float(np.dot(beta_norm, analytic_dir_norm))
    angle_diff_set_norm = minimal_angle_to_projection_set_deg(dicex_dir_norm, beta_norm, t_star)

    elapsed = time.perf_counter() - t0

    return {
        # Identification
        "dataset_name": artifact.dataset_name,
        "model_family": artifact.model_family,
        "task": artifact.task,
        "dimension_bucket": artifact.dimension_bucket,
        "n_features": d,
        "query_id": query_id,
        "preset": preset_name,
        # Noise configuration
        "alpha": noise_config.alpha,
        "noise_label": noise_label,
        "mu": noise_config.mu,
        "sigma": noise_config.sigma,
        "env_std": noise_config.env_std,
        # CVaR values: analytic → baseline → DiCEx
        "analytic_value": analytic_val,
        "analytic_cvar_norm_maxmin": analytic_val / (pred_range_test + epsilon_val),
        "baseline_random_cvar_median": baseline_random_cvar_median,
        "dicex_value": dicex_val,
        "dicex_value_analytic": dicex_val_analytic,
        "dicex_cvar_norm_maxmin": dicex_val / (pred_range_test + epsilon_val),
        # Value errors
        "val_diff": val_diff,
        "abs_val_diff": abs_val_diff,
        "rel_val_diff": rel_val_diff,
        "val_diff_direction": val_diff_direction,
        "rel_val_diff_direction": rel_val_diff_direction,
        "sampling_abs_val_diff": sampling_abs_val_diff,
        "sampling_rel_val_diff": sampling_rel_val_diff,
        "rel_val_diff_norm": rel_val_diff_norm,
        "baseline_improvement_norm": baseline_improvement_norm,
        "baseline_rel_error": baseline_rel_error,
        "rel_error_to_baseline": rel_error_to_baseline,
        # Angular errors
        "angle_diff_sol_norm_deg": angle_diff_sol_norm,
        "angle_diff_set_norm_deg": angle_diff_set_norm,
        "angle_diff_norm_deg": angle_diff_set_norm,
        # Hyperparameters
        "b_samples": b_samples,
        "n_exploration": n_exploration,
        "n_test": n_test,
        "n_evaluation": n_evaluation,
        "n_bootstrap": n_bootstrap,
        "eta": DICEX_CONFIG["eta"],
        "angular_schedule_deg": str(list(DICEX_CONFIG["angular_schedule_deg"])),
        "grid_t_len": DICEX_CONFIG["grid_t_len"],
        # Execution
        "elapsed_seconds": elapsed,
        "n_model_evals": dicex_res["metadata"].get("n_model_evals", 0),
    }


def _run_task(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Run a single comparison task in a worker process.

    Args:
        task_dict (dict[str, Any]): Dictionary with all arguments for run_single_comparison.

    Returns:
        dict[str, Any]: Dictionary with status, result (on success), or error details (on failure).
    """
    try:
        analytic_solutions = load_analytic_solutions()

        t = task_dict
        res = run_single_comparison(
            artifact=t["artifact"],
            x0=t["x0"],
            query_id=t["query_id"],
            noise_config=t["noise_config"],
            seed=t["seed"],
            analytic_solutions=analytic_solutions,
            pred_range_test=t["pred_range_test"],
            noise_label=t["label"],
            preset_name=t.get("preset_name", "high"),
            dicex_config=t.get("dicex_config"),
        )
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e), "t": task_dict}
    else:
        return {"status": "success", "res": res, "t": t}


def main() -> None:
    """Run the full analytic convergence grid comparison (CLI entry point).

    Loads trained linear/logistic models, computes query points, builds a grid
    over noise scenarios, alpha levels, and optimizer presets, executes comparisons
    in parallel, and saves the consolidated raw results CSV.
    """
    parser = argparse.ArgumentParser(description="Run analytical correctness study.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel processes (default: 1).")
    parser.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "minimal", "low", "mid", "high", "xhigh"],
        help="Optimizer presets to evaluate (default: all).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="all",
        choices=["low", "medium", "high", "all"],
        help="Dimension bucket to load artifacts for.",
    )
    parser.add_argument(
        "--model-families",
        type=str,
        nargs="+",
        default=["logistic", "linear"],
        choices=["logistic", "linear"],
        help="Model families to include (default: logistic linear).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Only run for these specific dataset(s) (space-separated names).",
    )
    args = parser.parse_args()

    seed = EXPERIMENT_CONFIG["seed"]
    layout = experiment_layout()
    ensure_layout(layout)
    log_dir = REPO_ROOT / "experiments" / "analytic_convergence" / "results" / "logs"
    configure_logging(layout, "analytic_convergence", log_dir=log_dir)

    # Every task compares against the closed-form optima of the "analytic" stage.
    solutions_path = get_solutions_path()
    if not solutions_path.exists():
        logger.error(
            "Precomputed analytical solutions not found at %s. Run `./run_all.sh analytic` first.",
            repo_relative(solutions_path),
        )
        sys.exit(1)

    # Create target folders
    results_dir = REPO_ROOT / "experiments" / "analytic_convergence" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    bucket_arg = None if args.bucket == "all" else args.bucket
    artifacts = load_trained_artifacts(layout, bucket=bucket_arg, model_families=args.model_families)

    # Filter by specific datasets if requested
    if args.datasets:
        ds_set = {d.strip() for d in args.datasets}
        before = len(artifacts)
        artifacts = [a for a in artifacts if a.dataset_name in ds_set]
        logger.info("Filtered to %d / %d artifacts matching datasets: %s", len(artifacts), before, sorted(ds_set))
        if not artifacts:
            logger.error("No artifacts match the requested datasets: %s", sorted(ds_set))
            sys.exit(1)

    logger.info("Computing query points...")
    compute_query_points_on_test(artifacts, tuple(EXPERIMENT_CONFIG["query_percentiles"]))

    tasks = []
    alphas = EXPERIMENT_CONFIG["alphas"]
    noise_scenarios = EXPERIMENT_CONFIG["noise_scenarios"]

    if "all" in args.presets:
        active_presets = ["minimal", "low", "mid", "high", "xhigh"]
    else:
        active_presets = args.presets

    logger.info("Preparing grid comparison tasks across presets: %s...", active_presets)

    # Precompute pred_range_test for each artifact
    artifact_ranges = {}
    for artifact in artifacts:
        predict_fn = get_predict_fn(artifact)
        test_preds = predict_fn(artifact.x_test)
        rng_val = float(np.max(test_preds) - np.min(test_preds))
        if rng_val < 1e-6:
            rng_val = 1.0
        artifact_ranges[artifact.artifact_id] = rng_val

    task_id = 0
    for preset_name in active_presets:
        preset_cfg = DICEX_PRESETS[preset_name]
        for artifact in artifacts:
            pred_range_test = artifact_ranges[artifact.artifact_id]
            for query_point in artifact.query_points:
                for scenario in noise_scenarios:
                    for alpha in alphas:
                        noise_config = NoiseRiskConfig(
                            mu=scenario["mu"],
                            sigma=scenario["sigma"],
                            env_std=scenario["env_std"],
                            alpha=alpha,
                        )
                        tasks.append(
                            {
                                "artifact": artifact,
                                "x0": query_point.x0,
                                "query_id": query_point.query_id,
                                "noise_config": noise_config,
                                "seed": seed + task_id,
                                "label": scenario["label"],
                                "pred_range_test": pred_range_test,
                                "preset_name": preset_name,
                                "dicex_config": preset_cfg,
                            }
                        )
                        task_id += 1

    logger.info("Total tasks scheduled: %d (Running in parallel with %d workers)", len(tasks), args.workers)

    results = []
    t_start = time.perf_counter()

    completed_count = 0
    with multiprocessing.Pool(args.workers) as pool:
        for output in pool.imap_unordered(_run_task, tasks):
            completed_count += 1
            t = output["t"]
            if output["status"] == "success":
                res = output["res"]
                results.append(res)
                logger.info(
                    "  [%d/%d] Completed %s (%d features, %s, alpha=%.2f) in %.2fs",
                    completed_count,
                    len(tasks),
                    t["artifact"].dataset_name,
                    len(t["x0"]),
                    t["label"],
                    t["noise_config"].alpha,
                    res["elapsed_seconds"],
                )
            else:
                logger.error(
                    "  [%d/%d] Error running comparison on %s (%s, alpha=%.2f): %s",
                    completed_count,
                    len(tasks),
                    t["artifact"].dataset_name,
                    t["label"],
                    t["noise_config"].alpha,
                    output["error"],
                )

    logger.info("Execution complete in %.2f seconds.", time.perf_counter() - t_start)

    df = pd.DataFrame(results)

    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    n_failed = len(tasks) - len(results)
    if n_failed:
        # The reporting scripts read the latest raw_results_*.csv, so an incomplete run is
        # saved under another prefix and never replaces the results of a complete run.
        incomplete_path = results_dir / f"incomplete_raw_results_{timestamp}.csv"
        df.to_csv(incomplete_path, index=False)
        logger.error(
            "%d/%d tasks failed; partial results saved to %s (ignored by the reporting scripts).",
            n_failed,
            len(tasks),
            repo_relative(incomplete_path),
        )
        sys.exit(1)

    raw_results_path = results_dir / f"raw_results_{timestamp}.csv"
    df.to_csv(raw_results_path, index=False)
    logger.info("Raw results saved to %s", repo_relative(raw_results_path))


if __name__ == "__main__":
    main()
