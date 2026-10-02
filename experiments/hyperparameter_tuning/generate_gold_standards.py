# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Generate the gold-standard reference values used by the hyperparameter tuning."""

from __future__ import annotations

from argparse import ArgumentParser
import multiprocessing
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from experiments.hyperparameter_tuning.config import default_gold_profile
from experiments.shared.analytic_solutions import load_analytic_solutions, make_solutions_key
from experiments.shared.evaluation import (
    analytic_reference_result,
    run_artifact_explanation,
)
from experiments.shared.helpers import angular_schedule_to_kappas, repo_relative
from experiments.shared.loader import load_trained_artifacts
from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.persistence import (
    ensure_layout,
    experiment_layout,
    gold_records_path,
    load_pickle,
    write_pickle,
)
from experiments.shared.query_points import compute_query_points_on_test

if TYPE_CHECKING:
    from experiments.hyperparameter_tuning.config import GoldProfile
    from experiments.shared.bundles import TrainedModelArtifact

logger = get_logger("generate_gold_standards")


def _process_gold_task(task: dict[str, Any]) -> dict[str, Any]:
    """Execute gold record generation for a single combination in a worker process.

    Args:
        task (dict[str, Any]): Dictionary containing task configuration:
            artifact, query_point, noise_config, profile, and local_seed.

    Returns:
        dict[str, Any]: Dictionary containing generated gold record metadata and results.
    """
    artifact = task["artifact"]
    query_point = task["query_point"]
    noise_config = task["noise_config"]
    profile = task["profile"]
    local_seed = task["local_seed"]

    d = int(artifact.x_train.shape[1])
    bucket_budget = profile.budgets_by_bucket[artifact.dimension_bucket]

    analytic_record = analytic_reference_result(artifact, x0=query_point.x0, noise_config=noise_config)

    if analytic_record is None:
        kappas = angular_schedule_to_kappas(profile.angular_schedule_deg, d)
        optimizer_kwargs = {
            "kappas": kappas,
            "b_samples": profile.b_samples,
            "n_exploration": profile.n_exploration,
            "n_test": profile.n_test,
            "n_evaluation": profile.n_evaluation,
            "n_bootstrap": profile.n_bootstrap,
            "eta": profile.eta,
            "grid_t": np.linspace(0.0, 1.0, profile.grid_t_len),
            "max_evals": None,
            "trace_level": "none",
            "min_runtime_seconds": bucket_budget.min_runtime_seconds,
            "timeout_seconds": bucket_budget.min_runtime_seconds,
            "seed": local_seed,
        }
        logger.info(
            "  [deep_search] query: %s (p=%s) | noise: %s (env_std=%s, alpha=%s)",
            query_point.query_id,
            query_point.percentile,
            noise_config.mu,
            noise_config.env_std,
            noise_config.alpha,
        )
        runtime_record = run_artifact_explanation(
            artifact,
            x0=query_point.x0,
            noise_config=noise_config,
            optimizer_kwargs=optimizer_kwargs,
        )
        reference_kind = "deep_search"
    else:
        logger.info(
            "  [analytic] query: %s (p=%s) | noise: %s (env_std=%s, alpha=%s)",
            query_point.query_id,
            query_point.percentile,
            noise_config.mu,
            noise_config.env_std,
            noise_config.alpha,
        )
        runtime_record = analytic_record
        optimizer_kwargs = {}
        reference_kind = str(analytic_record["reference_kind"])

    return {
        "artifact_id": artifact.artifact_id,
        "dataset_name": artifact.dataset_name,
        "task": artifact.task,
        "model_family": artifact.model_family,
        "dimension_bucket": artifact.dimension_bucket,
        "query_id": query_point.query_id,
        "query_percentile": float(query_point.percentile),
        "point_index": int(query_point.point_index),
        "point_value": float(query_point.point_value),
        "noise_config": noise_config.to_dict(),
        "reference_kind": reference_kind,
        "gold_value": float(runtime_record["robust_value"]),
        "direction": [float(v) for v in runtime_record["direction"]],
        "metadata": {
            "n_model_evals": int(runtime_record["metadata"].get("n_model_evals", 0)),
            "stop_reason": str(runtime_record["metadata"].get("stop_reason", "unknown")),
            "wall_time_seconds": float(runtime_record["metadata"].get("wall_time_seconds", 0.0)),
        },
        "optimizer_config": optimizer_kwargs,
    }


def generate_gold_records(
    profile: GoldProfile,
    artifacts: list[TrainedModelArtifact],
    dest_path: Path,
    *,
    seed: int,
    workers: int = 1,
    existing_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate one gold-standard record per artifact/query/noise configuration.

    Args:
        profile (GoldProfile): Configuration profile for gold standard generation.
        artifacts (list[TrainedModelArtifact]): List of trained model artifacts to evaluate.
        dest_path (Path): Destination file path for saving progress.
        seed (int): Base random seed for reproducibility.
        workers (int): Number of worker processes for parallel execution.
        existing_records (list[dict[str, Any]] | None): Optional list of previously computed records.

    Returns:
        list[dict[str, Any]]: List of generated or loaded gold-standard records.
    """
    logger.info(
        "Starting gold-standard generation for %d model artifact(s) (seed=%d, workers=%d)...",
        len(artifacts),
        seed,
        workers,
    )

    # Build a lookup map for existing records
    existing_map: dict[tuple[str, str, float, float, float, float], dict[str, Any]] = {}
    if existing_records:
        for r in existing_records:
            key = make_solutions_key(r["artifact_id"], r["query_id"], r["noise_config"])
            existing_map[key] = r

    # The gold standards of linear and logistic models are their closed-form optima.
    try:
        analytic_sol_map = load_analytic_solutions()
    except FileNotFoundError:
        logger.error("Precomputed analytical solutions not found. Run `./run_all.sh analytic` first.")
        sys.exit(1)

    records: list[dict[str, Any]] = []
    if existing_records:
        active_artifact_ids = {a.artifact_id for a in artifacts}
        records.extend(r for r in existing_records if r["artifact_id"] not in active_artifact_ids)

    tasks_to_run = []

    for artifact_index, artifact in enumerate(artifacts):
        d = int(artifact.x_train.shape[1])
        logger.info(
            "Preparing tasks for artifact '%s' (bucket: %s, %d features, %d query points)",
            artifact.artifact_id,
            artifact.dimension_bucket,
            d,
            len(artifact.query_points),
        )
        for query_index, query_point in enumerate(artifact.query_points):
            for noise_index, noise_config in enumerate(profile.parameter_grid):
                key = make_solutions_key(artifact.artifact_id, query_point.query_id, noise_config)
                if key in existing_map:
                    logger.info(
                        "  [skip] query: %s | noise: mu=%s (Already computed, resuming)",
                        query_point.query_id,
                        noise_config.mu,
                    )
                    records.append(existing_map[key])
                    continue

                if key in analytic_sol_map:
                    logger.info(
                        "  [analytic_cache] query: %s | noise: mu=%s (Retrieved from analytic_solutions.pkl)",
                        query_point.query_id,
                        noise_config.mu,
                    )
                    sol = analytic_sol_map[key]
                    records.append(
                        {
                            "artifact_id": artifact.artifact_id,
                            "dataset_name": artifact.dataset_name,
                            "task": artifact.task,
                            "model_family": artifact.model_family,
                            "dimension_bucket": artifact.dimension_bucket,
                            "query_id": query_point.query_id,
                            "query_percentile": float(query_point.percentile),
                            "point_index": int(query_point.point_index),
                            "point_value": float(query_point.point_value),
                            "noise_config": noise_config.to_dict(),
                            "reference_kind": "analytic",
                            "gold_value": float(sol["analytic_val"]),
                            "direction": [float(v) for v in sol["analytic_dir_norm"]],
                            "metadata": {
                                "n_model_evals": 0,
                                "stop_reason": "analytic",
                                "wall_time_seconds": 0.0,
                            },
                            "optimizer_config": {},
                        }
                    )
                    continue

                local_seed = seed + artifact_index * 10_000 + query_index * 100 + noise_index
                tasks_to_run.append(
                    {
                        "artifact": artifact,
                        "query_point": query_point,
                        "noise_config": noise_config,
                        "profile": profile,
                        "local_seed": local_seed,
                    }
                )

    if tasks_to_run:
        logger.info("Running %d tasks with %d worker processes...", len(tasks_to_run), workers)
        if workers > 1:
            with multiprocessing.Pool(workers) as pool:
                for new_record in pool.imap_unordered(_process_gold_task, tasks_to_run):
                    records.append(new_record)
                    # Persist immediately to disk
                    persist_gold_records(dest_path, records, seed=seed, verbose=False)
        else:
            for task in tasks_to_run:
                new_record = _process_gold_task(task)
                records.append(new_record)
                persist_gold_records(dest_path, records, seed=seed, verbose=False)

    logger.info("Generated %d gold records.", len(records))
    return records


def persist_gold_records(
    dest_path: Path, records: list[dict[str, Any]], *, seed: int, verbose: bool = True
) -> dict[str, Any]:
    """Persist the generated gold records to the gold-standard pickle file.

    Args:
        dest_path (Path): Destination filesystem path for the pickle file.
        records (list[dict[str, Any]]): List of gold record dictionaries to persist.
        seed (int): Random seed used during generation.
        verbose (bool): Whether to log informational messages during persistence.

    Returns:
        dict[str, Any]: Payload dictionary stored to disk containing seed, count, and records.
    """
    if verbose:
        logger.info("Persisting %d gold records to %s", len(records), repo_relative(dest_path))
    payload = {
        "seed": seed,
        "n_records": len(records),
        "records": records,
    }
    write_pickle(dest_path, payload)
    if verbose:
        logger.info("Done persisting gold records.")
    return payload


def main() -> None:
    """Generate the gold-standard reference artifact (CLI entry point).

    Raises:
        RuntimeError: If no trained model artifacts match the filter criteria.
    """
    parser = ArgumentParser(description="Generate the gold-standard reference artifact.")
    parser.add_argument("--model-family", type=str, default=None, help="Filter to single model family.")
    parser.add_argument("--bucket", type=str, default=None, help="Filter to single dimension bucket (e.g. low).")
    parser.add_argument("--dataset", type=str, nargs="*", default=None, help="Filter to specific dataset name(s).")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix for output file (e.g. xgboost_low).")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers for multiprocessing (default: 1).")
    args = parser.parse_args()

    seed = 20260530
    profile = default_gold_profile()

    layout = experiment_layout()
    ensure_layout(layout)
    configure_logging(layout, "generate_gold_standards")

    dest_path = (
        layout.gold_root / f"gold_standards_{args.output_suffix}.pkl"
        if args.output_suffix
        else gold_records_path(layout)
    )

    logger.info("Starting gold-standards generation (seed=%d)", seed)
    artifacts = load_trained_artifacts(layout)

    excluded_datasets = {
        "tecator",
        "ailerons",
        "sonar",
        "eeg-eye-state",
        "sulfur",
        "blood_transfusion",
    }
    artifacts = [a for a in artifacts if a.dataset_name not in excluded_datasets]

    if args.model_family:
        artifacts = [a for a in artifacts if a.model_family == args.model_family]
    if args.bucket:
        artifacts = [a for a in artifacts if a.dimension_bucket == args.bucket]
    if args.dataset:
        artifacts = [a for a in artifacts if a.dataset_name in args.dataset]

    if not artifacts:
        msg = "No trained model artifacts found matching the filter criteria."
        logger.error(msg)
        raise RuntimeError(msg)

    # Compute query points on test set for each artifact independently
    compute_query_points_on_test(artifacts, profile.query_percentiles)

    # Load existing gold standards if they exist and seed matches
    existing_records: list[dict[str, Any]] = []
    if dest_path.exists():
        try:
            existing_payload = load_pickle(dest_path)
            if existing_payload.get("seed") == seed:
                existing_records = existing_payload.get("records", [])
                logger.info(
                    "Found existing gold records file with matching seed (%d records). Resuming...",
                    len(existing_records),
                )
            else:
                logger.info(
                    "Found existing gold records file but seed mismatch (current: %d, file: %s). Starting fresh.",
                    seed,
                    existing_payload.get("seed"),
                )
        except (OSError, KeyError, ValueError, RuntimeError) as e:
            logger.warning("Could not read existing gold records file (%s). Starting fresh.", e)

    records = generate_gold_records(
        profile, artifacts, dest_path, seed=seed, workers=args.workers, existing_records=existing_records
    )
    payload = persist_gold_records(dest_path, records, seed=seed, verbose=True)

    logger.info("Saved %d gold-standard records to %s.", payload["n_records"], repo_relative(dest_path))


if __name__ == "__main__":
    main()
