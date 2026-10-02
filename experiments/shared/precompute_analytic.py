# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Precompute and cache the closed-form solutions used by the analytic-convergence study."""

from __future__ import annotations

from argparse import ArgumentParser
import multiprocessing
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from dicex.analytical.linear import linear_directional_functional, linear_optimal_direction
from dicex.analytical.logistic import logistic_directional_functional, logistic_optimal_direction
from dicex.core.functional import directional_improvement
from dicex.distributions.gaussian import GaussianPerturbation
from experiments.shared.analytic_solutions import load_analytic_solutions, make_solutions_key, save_analytic_solutions
from experiments.shared.loader import load_trained_artifacts
from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.persistence import ensure_layout, experiment_layout
from experiments.shared.query_points import compute_query_points_on_test
from experiments.shared.specs import NoiseRiskConfig

if TYPE_CHECKING:
    from experiments.shared.bundles import TrainedModelArtifact

logger = get_logger("precompute_analytic")

CONFIG_PATH = REPO_ROOT / "experiments" / "analytic_convergence" / "experiment_config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    _FULL_CONFIG = yaml.safe_load(f)

ALPHAS = _FULL_CONFIG["experiment"]["alphas"]
NOISE_SCENARIOS = _FULL_CONFIG["experiment"]["noise_scenarios"]
QUERY_PERCENTILES = tuple(_FULL_CONFIG["experiment"]["query_percentiles"])
SIGMA_Z_SAMPLES: int = 1_000_000


def _estimate_sigma_z(
    model: Any,
    x0: np.ndarray,
    c: np.ndarray,
    n: int,
    perturbation: GaussianPerturbation,
    target_class: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    r"""Estimate :math:`\sigma_Z = \operatorname{std}[Z(c)]` empirically.

    Args:
        model (Any): Predictor model or ModelScorer instance.
        x0 (np.ndarray): Baseline point (1D array).
        c (np.ndarray): Direction vector (1D array).
        n (int): Number of perturbation samples.
        perturbation (GaussianPerturbation): Perturbation distribution specs.
        target_class (int | None): Optional index for classification models.
        rng (np.random.Generator | None): Optional random number generator.

    Returns:
        float: Empirical standard deviation of improvement samples.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = perturbation.sample(n, d=len(x0), rng=rng)
    env_noise = perturbation.sample_env(n, len(x0), rng=rng)
    z_vals = directional_improvement(model, x0, c, noise, env_noise, target_class)
    return float(np.std(z_vals, ddof=1))


def compute_single_analytical(
    artifact: TrainedModelArtifact, x0: np.ndarray, noise_config: NoiseRiskConfig
) -> dict[str, Any] | None:
    """Compute the analytical solution and normalized values for caching.

    Args:
        artifact (TrainedModelArtifact): Reconstructed trained-model bundle.
        x0 (np.ndarray): Baseline query point.
        noise_config (NoiseRiskConfig): Perturbation noise parameters.

    Returns:
        dict[str, Any] | None: Dictionary containing analytical direction, value, and metadata,
            or None if non-analytic.
    """
    env_cov = np.full(len(x0), noise_config.env_std**2, dtype=float)
    perturbation = GaussianPerturbation(mu=noise_config.mu, sigma=noise_config.sigma, env_cov=env_cov)

    if artifact.model_family == "linear":
        coef = artifact.model.coef_
        beta_raw = np.asarray(coef, dtype=float).reshape(-1)
        if artifact.scaler is not None:
            beta_norm = beta_raw
            x0_eval = artifact.scaler.transform(np.asarray(x0, dtype=float).reshape(1, -1)).reshape(-1)
            intercept_norm = 0.0  # Not used by the linear directional functional
        else:
            mean = np.mean(artifact.x_train, axis=0)
            std = np.std(artifact.x_train, axis=0)
            std[std < 1e-8] = 1.0
            beta_norm = beta_raw * std
            x0_eval = (np.asarray(x0, dtype=float) - mean) / std
            intercept_norm = 0.0

        analytic_dir_norm = linear_optimal_direction(
            beta_norm,
            perturbation,
            noise_config.alpha,
            n_refinement_passes=2,
        )
        analytic_val = linear_directional_functional(beta_norm, perturbation, noise_config.alpha, analytic_dir_norm)

    elif artifact.model_family == "logistic":
        coef = artifact.model.coef_
        intercept = artifact.model.intercept_
        if coef.shape[0] > 1:
            if artifact.scaler is not None:
                x_scaled = artifact.scaler.transform(artifact.x_train)
            else:
                x_scaled = artifact.x_train
            probs = artifact.model.predict_proba(x_scaled)
            variances = np.var(probs, axis=0)
            class_idx = int(np.argmax(variances))
            beta_class = coef[class_idx]
            intercept_class = intercept[class_idx]
        else:
            beta_class = coef[0]
            intercept_class = intercept[0]

        beta_raw = np.asarray(beta_class, dtype=float).reshape(-1)
        intercept_raw = float(np.asarray(intercept_class, dtype=float).reshape(-1)[0])

        if artifact.scaler is not None:
            beta_norm = beta_raw
            intercept_norm = intercept_raw
            x0_eval = artifact.scaler.transform(np.asarray(x0, dtype=float).reshape(1, -1)).reshape(-1)
        else:
            mean = np.mean(artifact.x_train, axis=0)
            std = np.std(artifact.x_train, axis=0)
            std[std < 1e-8] = 1.0
            beta_norm = beta_raw * std
            intercept_norm = intercept_raw + np.dot(beta_raw, mean)
            x0_eval = (np.asarray(x0, dtype=float) - mean) / std

        analytic_dir_norm = logistic_optimal_direction(
            beta_norm,
            intercept_norm,
            x0_eval,
            perturbation,
            noise_config.alpha,
            n_refinement_passes=2,
        )
        analytic_val = logistic_directional_functional(
            beta_norm, intercept_norm, x0_eval, perturbation, noise_config.alpha, analytic_dir_norm
        )

    else:
        return None

    t_opt_norm = float(np.dot(beta_norm, analytic_dir_norm))

    # sigma_Z: empirical standard deviation of Z(c) for the optimal direction
    rng_sigma = np.random.default_rng(42)
    sigma_z = _estimate_sigma_z(
        model=artifact.model,
        x0=x0_eval,
        c=analytic_dir_norm,
        n=SIGMA_Z_SAMPLES,
        perturbation=perturbation,
        target_class=artifact.target_class,
        rng=rng_sigma,
    )

    return {
        "analytic_dir_norm": np.asarray(analytic_dir_norm, dtype=float),
        "analytic_val": float(analytic_val),
        "beta_norm": np.asarray(beta_norm, dtype=float),
        "intercept_norm": float(intercept_norm),
        "x0_eval": np.asarray(x0_eval, dtype=float),
        "t_opt_norm": float(t_opt_norm),
        "sigma_z": sigma_z,
    }


def _process_task(task: dict[str, Any]) -> dict[str, Any]:
    """Compute a single cache entry in a worker process.

    Args:
        task (dict[str, Any]): Task payload containing artifact, query point, and noise config.

    Returns:
        dict[str, Any]: Dictionary containing solution key and precomputed result.
    """
    artifact = task["artifact"]
    qp = task["qp"]
    noise_config = task["noise_config"]
    res = compute_single_analytical(artifact, qp.x0, noise_config)

    key = make_solutions_key(artifact.artifact_id, str(qp.query_id), noise_config)
    return {"key": key, "result": res}


def main() -> None:
    """Run the precomputation pipeline for analytical solutions (CLI entry point)."""
    parser = ArgumentParser(description="Precompute analytical solutions for experiments.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1).")
    parser.add_argument(
        "--bucket",
        type=str,
        default="all",
        choices=["low", "medium", "high", "all"],
        help="Dimension bucket to process.",
    )
    parser.add_argument(
        "--sigma-z-samples",
        type=int,
        default=1_000_000,
        help="Samples for the empirical sigma_Z estimate (default: 1000000).",
    )
    parser.add_argument(
        "--force-datasets",
        nargs="*",
        default=None,
        help=(
            "Force recompute for specific dataset(s), even if cached. "
            "Format: 'dataset_name' or 'dataset_name:model_family'. "
            "Use 'all' to force everything. Examples:\n"
            "  --force-datasets tecator\n"
            "  --force-datasets buzz_toms_hardware:linear\n"
            "  --force-datasets ailerons elevators\n"
            "  --force-datasets all\n"
            "  --force-datasets all:linear\n"
            "  --force-datasets all:logistic"
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Only process these specific dataset(s) (space-separated names).",
    )
    args = parser.parse_args()
    global SIGMA_Z_SAMPLES  # noqa: PLW0603
    SIGMA_Z_SAMPLES = args.sigma_z_samples

    layout = experiment_layout()
    ensure_layout(layout)
    configure_logging(layout, "precompute_analytic")

    logger.info("Loading models...")
    bucket_arg = None if args.bucket == "all" else args.bucket
    artifacts = load_trained_artifacts(layout, bucket=bucket_arg, model_families=["logistic", "linear"])
    logger.info("Loaded %d artifacts.", len(artifacts))

    # Filter by specific datasets if requested
    if args.datasets:
        ds_set = {d.strip() for d in args.datasets}
        before = len(artifacts)
        artifacts = [a for a in artifacts if a.dataset_name in ds_set]
        logger.info("Filtered to %d / %d artifacts matching datasets: %s", len(artifacts), before, sorted(ds_set))
        if not artifacts:
            logger.error("No artifacts match the requested datasets: %s", sorted(ds_set))
            sys.exit(1)

    logger.info("Computing query points at test set...")
    compute_query_points_on_test(artifacts, QUERY_PERCENTILES)

    try:
        existing_solutions = load_analytic_solutions()
        logger.info("Loaded existing solutions with %d entries.", len(existing_solutions))
    except FileNotFoundError:
        existing_solutions = {}
        logger.info("No existing solutions found. Starting fresh.")

    # Parse --force-datasets
    force_all = False
    # (dataset_name, model_family_or_None) — None means all families for that dataset
    force_set: set[tuple[str, str | None]] = set()
    # Model-wide force: e.g. "all:linear" drops all linear entries
    force_model_set: set[str] = set()
    if args.force_datasets is not None:
        for entry in args.force_datasets:
            entry_stripped = entry.strip()
            if entry_stripped == "all":
                force_all = True
            elif entry_stripped.startswith("all:") and entry_stripped.count(":") == 1:
                force_model_set.add(entry_stripped.split(":", 1)[1].strip())
            elif ":" in entry_stripped:
                ds, mf = entry_stripped.split(":", 1)
                force_set.add((ds.strip(), mf.strip()))
            else:
                force_set.add((entry_stripped, None))

    if force_all:
        logger.info("Forcing recomputation of all tasks (--force-datasets all)")
        existing_solutions = {}
    elif force_model_set:
        logger.info("Force-recompute for model(s): %s", force_model_set)
        keys_to_drop: list[tuple[str, str, float, float, float, float]] = []
        for key in list(existing_solutions.keys()):
            artifact_id: str = key[0]
            mf_name = artifact_id.split("__")[1] if "__" in artifact_id else ""
            if mf_name in force_model_set:
                keys_to_drop.append(key)
        for k in keys_to_drop:
            del existing_solutions[k]
        logger.info("Dropped %d cached entries for model(s).", len(keys_to_drop))
    elif force_set:
        logger.info("Force-recompute for %d dataset(s): %s", len(force_set), force_set)
        # Each key is a tuple (artifact_id, query_id, alpha, mu, sigma, env_std), where
        # artifact_id has the form "<dataset_name>__<model_family>".
        keys_to_drop: list[tuple[str, str, float, float, float, float]] = []
        for key in list(existing_solutions.keys()):
            artifact_id: str = key[0]
            ds_name = artifact_id.split("__")[0]
            mf_name = artifact_id.split("__")[1] if "__" in artifact_id else ""
            for force_ds, force_mf in force_set:
                if force_ds == ds_name and (force_mf is None or force_mf == mf_name):
                    keys_to_drop.append(key)
                    break
        for k in keys_to_drop:
            del existing_solutions[k]
        logger.info("Dropped %d cached entries for forced datasets.", len(keys_to_drop))

    tasks = []
    for artifact in artifacts:
        logger.info("Processing artifact: %s / %s", artifact.dataset_name, artifact.model_family)
        for qp in artifact.query_points:
            for scenario in NOISE_SCENARIOS:
                for alpha in ALPHAS:
                    noise_config = NoiseRiskConfig(
                        mu=scenario["mu"],
                        sigma=scenario["sigma"],
                        env_std=scenario["env_std"],
                        alpha=alpha,
                    )
                    key = make_solutions_key(artifact.artifact_id, str(qp.query_id), noise_config)
                    if key in existing_solutions:
                        continue
                    tasks.append(
                        {
                            "artifact": artifact,
                            "qp": qp,
                            "noise_config": noise_config,
                        }
                    )

    logger.info("Prepared %d tasks for precomputation. Executing with %d workers...", len(tasks), args.workers)

    solutions = {}
    t0 = time.perf_counter()

    with multiprocessing.Pool(args.workers) as pool:
        for i, outcome in enumerate(pool.imap_unordered(_process_task, tasks)):
            if outcome["result"] is not None:
                solutions[outcome["key"]] = outcome["result"]

            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                logger.info("Progress: %d / %d tasks completed.", i + 1, len(tasks))

    logger.info("Precomputation finished in %.2fs. Computed %d solutions.", time.perf_counter() - t0, len(solutions))

    save_analytic_solutions(solutions)
    logger.info("Analytic solutions saved.")


if __name__ == "__main__":
    main()
