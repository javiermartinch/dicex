# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Multi-objective (NSGA-II) hyperparameter tuning of the DiCEx optimizer per dimension bucket."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime
import json
import math
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
import optuna
from optuna.storages import JournalFileStorage, JournalStorage

from experiments.hyperparameter_tuning.config import default_optuna_profile
from experiments.shared.evaluation import (
    run_artifact_explanation,
)
from experiments.shared.helpers import angular_schedule_to_kappas
from experiments.shared.loader import load_trained_artifacts
from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.persistence import (
    ensure_layout,
    experiment_layout,
    gold_records_path,
    load_pickle,
    optuna_study_path,
    write_json,
    write_yaml,
)
from experiments.shared.query_points import compute_query_points_on_test
from experiments.shared.specs import DIMENSION_BUCKET_ORDER, NoiseRiskConfig

if TYPE_CHECKING:
    from experiments.hyperparameter_tuning.config import OptunaProfile
    from experiments.shared.bundles import TrainedModelArtifact
    from experiments.shared.persistence import RunLayout
    from experiments.shared.specs import DimensionBucket

logger = get_logger("hyperparameter_tuning")

_worker_cases: list[GoldEvaluationCase] | None = None


def _init_worker(cases: list[GoldEvaluationCase]) -> None:
    """Initialize the worker process with the list of gold evaluation cases.

    Args:
        cases (list[GoldEvaluationCase]): List of gold evaluation cases available to the worker.
    """
    global _worker_cases  # noqa: PLW0603
    _worker_cases = cases


def _process_optuna_case(task: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single gold case inside a worker process.

    Args:
        task (dict[str, Any]): Dictionary containing case index and optimizer arguments.

    Returns:
        dict[str, Any]: Evaluation result containing loss, metrics, and stop reason.

    Raises:
        RuntimeError: If worker process was not initialized with evaluation cases.
    """
    if _worker_cases is None:
        msg = "Worker process not initialized with cases."
        raise RuntimeError(msg)

    case_index = task["case_index"]
    optimizer_kwargs = task["optimizer_kwargs"]
    case = _worker_cases[case_index]

    run_record = run_artifact_explanation(
        case.artifact,
        x0=case.x0,
        noise_config=case.noise_config,
        optimizer_kwargs=optimizer_kwargs,
    )
    metadata = run_record["metadata"]
    loss = float(abs(run_record["robust_value"] - case.gold_value) / (abs(case.gold_value) + 1e-6))
    wall_time = float(metadata.get("wall_time_seconds", 0.0))
    n_evals = float(metadata.get("n_model_evals", 0))
    stop_reason = metadata.get("stop_reason", "unknown")

    return {
        "case_index": case_index,
        "loss": loss,
        "robust_value": run_record["robust_value"],
        "gold_value": case.gold_value,
        "wall_time_seconds": wall_time,
        "n_model_evals": n_evals,
        "stop_reason": stop_reason,
    }


@dataclass(frozen=True)
class GoldEvaluationCase:
    """One joined case combining a trained artifact, a query point, and its gold value."""

    artifact: TrainedModelArtifact
    query_id: str
    x0: np.ndarray
    noise_config: NoiseRiskConfig
    gold_value: float


def run_all_bucket_studies(
    profile: OptunaProfile,
    layout: RunLayout,
    *,
    seed: int,
    bucket: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the three independent Optuna studies and persist their artifacts.

    Args:
        profile (OptunaProfile): Optuna configuration profile.
        layout (RunLayout): File layout and directory configuration.
        seed (int): Base random seed.
        bucket (str | None): Specific dimension bucket to run, or None for all.
        workers (int): Number of worker processes.

    Returns:
        dict[str, Any]: Dictionary containing study summaries and the report path.
    """
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")

    logger.info("Starting Optuna search pipeline (seed=%d, bucket=%s)...", seed, bucket)
    artifacts = load_trained_artifacts(layout, bucket=bucket)

    excluded_datasets = {
        "tecator",
        "ailerons",
        "sonar",
        "eeg-eye-state",
        "sulfur",
        "blood_transfusion",
    }
    artifacts = [a for a in artifacts if a.dataset_name not in excluded_datasets]

    # Compute query points on test set for each artifact independently
    compute_query_points_on_test(artifacts, profile.query_percentiles)

    gold_payload = load_pickle(gold_records_path(layout))
    records = gold_payload.get("records", [])
    if bucket is not None:
        allowed_ids = {a.artifact_id for a in artifacts}
        records = [r for r in records if r["artifact_id"] in allowed_ids]
    cases = build_gold_cases(artifacts, records)
    logger.info("Loaded %d gold cases across all artifacts.", len(cases))

    buckets_to_run = [bucket] if bucket else DIMENSION_BUCKET_ORDER

    bucket_summaries: list[dict[str, Any]] = []
    for bucket_index, b in enumerate(buckets_to_run):
        bucket_cases = [case for case in cases if case.artifact.dimension_bucket == b]
        if not bucket_cases:
            logger.info("No gold cases found for bucket '%s'; skipping.", b)
            continue
        logger.info(
            "Running multi-objective study for bucket '%s' (%d cases, %d trials requested)...",
            b,
            len(bucket_cases),
            profile.trials_by_bucket[b],
        )
        study = _run_bucket_study(
            profile,
            layout,
            b,
            bucket_cases,
            seed=seed + bucket_index * 100_000,
            timestamp=timestamp,
            workers=workers,
        )
        selected_trial, pareto_trials, trial_rows = _summarize_study(study)

        # Sort to find low, mid, and high budget configurations on the Pareto frontier
        pareto_by_time = sorted(pareto_trials, key=lambda item: (item["mean_time_seconds"], item["precision_loss"]))
        low_trial = pareto_by_time[0]

        mid_trial = selected_trial

        pareto_by_loss = sorted(pareto_trials, key=lambda item: (item["precision_loss"], item["mean_time_seconds"]))
        high_trial = pareto_by_loss[0]

        low_payload = _selected_config_payload(b, low_trial, profile)
        mid_payload = _selected_config_payload(b, mid_trial, profile)
        high_payload = _selected_config_payload(b, high_trial, profile)

        config_path = layout.selected_configs_root / f"{b}_recommended.yaml"
        write_yaml(layout.selected_configs_root / f"{b}_low.yaml", low_payload)
        write_yaml(layout.selected_configs_root / f"{b}_mid.yaml", mid_payload)
        write_yaml(layout.selected_configs_root / f"{b}_high.yaml", high_payload)
        write_yaml(config_path, mid_payload)
        summary = {
            "bucket": b,
            "n_cases": len(bucket_cases),
            "n_trials_requested": profile.trials_by_bucket[b],
            "selected_trial": selected_trial,
            "pareto_trials": pareto_trials,
            "trial_rows": trial_rows,
            "config_path": str(config_path),
        }
        study_path = optuna_study_path(layout, b)
        logger.info(
            "Completed bucket '%s' study. Best trial: #%d (loss=%.6f, time=%.4fs). Saving to %s",
            b,
            selected_trial["trial_number"],
            selected_trial["precision_loss"],
            selected_trial["mean_time_seconds"],
            study_path,
        )
        write_json(study_path, summary)
        bucket_summaries.append(summary)

    logger.info("Optuna search pipeline complete. Run visualize_and_report.py to generate the report.")
    return {
        "studies": bucket_summaries,
    }


def build_gold_cases(artifacts: list[TrainedModelArtifact], records: list[dict[str, Any]]) -> list[GoldEvaluationCase]:
    """Join persisted gold records with the in-memory training artifacts.

    Args:
        artifacts (list[TrainedModelArtifact]): List of trained model artifacts.
        records (list[dict[str, Any]]): List of persisted gold standard records.

    Returns:
        list[GoldEvaluationCase]: List of constructed gold evaluation case objects.
    """
    artifact_map = {artifact.artifact_id: artifact for artifact in artifacts}

    cases: list[GoldEvaluationCase] = []
    for record in records:
        artifact = artifact_map[record["artifact_id"]]
        query_point = next(query for query in artifact.query_points if query.query_id == record["query_id"])
        noise_dict = record["noise_config"]
        noise_config = NoiseRiskConfig(
            mu=float(noise_dict["mu"]),
            sigma=float(noise_dict["sigma"]),
            env_std=float(noise_dict["env_std"]),
            alpha=float(noise_dict["alpha"]),
        )
        cases.append(
            GoldEvaluationCase(
                artifact=artifact,
                query_id=str(record["query_id"]),
                x0=np.asarray(query_point.x0, dtype=float),
                noise_config=noise_config,
                gold_value=float(record["gold_value"]),
            )
        )
    return cases


def _run_bucket_study(
    profile: OptunaProfile,
    layout: RunLayout,
    bucket: DimensionBucket,
    cases: list[GoldEvaluationCase],
    *,
    seed: int,
    timestamp: str,
    workers: int,
) -> Any:
    """Execute an Optuna study for a specific dimensionality bucket.

    Args:
        profile (OptunaProfile): Optuna configuration profile.
        layout (RunLayout): File layout and directory configuration.
        bucket (DimensionBucket): Dimensionality bucket name.
        cases (list[GoldEvaluationCase]): Gold evaluation cases for this bucket.
        seed (int): Random seed for the study.
        timestamp (str): Formatted timestamp string for naming.
        workers (int): Number of worker processes.

    Returns:
        optuna.Study: Completed Optuna study object.
    """
    storage_path = layout.optuna_root / f"{bucket}_journal.log"
    storage = JournalStorage(JournalFileStorage(str(storage_path)))

    sampler = optuna.samplers.NSGAIISampler(seed=seed, population_size=max(4, profile.trials_by_bucket[bucket]))
    study = optuna.create_study(
        study_name=f"{bucket}_study_{timestamp}",
        storage=storage,
        directions=["minimize", "minimize"],
        sampler=sampler,
        load_if_exists=True,
    )

    # Persist every completed trial to a timestamped JSONL file
    jsonl_path = layout.optuna_root / f"{bucket}_trials_{timestamp}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Start from an empty file
    if jsonl_path.exists():
        jsonl_path.unlink()

    def save_trial_callback(_study: Any, trial: Any) -> None:
        if trial.state.name != "COMPLETE" or trial.values is None:
            return
        trial_data = {
            "trial_number": trial.number,
            "precision_loss": float(trial.values[0]),
            "mean_time_seconds": float(trial.values[1]),
            "mean_n_model_evals": float(trial.user_attrs.get("summary", {}).get("mean_n_model_evals", 0.0)),
            "config": dict(trial.user_attrs.get("config_payload", {})),
            "cases": list(trial.user_attrs.get("cases", [])),
        }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trial_data) + "\n")

    if workers > 1:
        logger.info("Initializing multiprocessing pool with %d workers...", workers)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(cases,)) as pool:
            study.optimize(
                _objective(profile, bucket, cases, seed=seed, pool=pool, _workers=workers),
                n_trials=profile.trials_by_bucket[bucket],
                show_progress_bar=False,
                callbacks=[save_trial_callback],
            )
    else:
        logger.info("Running sequentially...")
        _init_worker(cases)
        study.optimize(
            _objective(profile, bucket, cases, seed=seed, pool=None, _workers=1),
            n_trials=profile.trials_by_bucket[bucket],
            show_progress_bar=False,
            callbacks=[save_trial_callback],
        )
    return study


def _objective(
    profile: OptunaProfile,
    bucket: DimensionBucket,
    cases: list[GoldEvaluationCase],
    *,
    seed: int,
    pool: multiprocessing.pool.Pool | None,
    _workers: int,
):
    """Build the multi-objective optimization function for an Optuna trial.

    Args:
        profile (OptunaProfile): Optuna configuration profile.
        bucket (DimensionBucket): Dimensionality bucket name.
        cases (list[GoldEvaluationCase]): Gold evaluation cases to optimize against.
        seed (int): Base random seed.
        pool (multiprocessing.pool.Pool | None): Worker pool for parallel evaluation, or None.
        _workers (int): Number of worker processes.

    Returns:
        Callable[[optuna.Trial], tuple[float, float]]: Multi-objective evaluation function.
    """
    timeout_seconds = profile.timeouts_by_bucket[bucket]

    def objective(trial: Any) -> tuple[float, float]:
        search_space = _search_space(profile)
        n_vecindarios = trial.suggest_int("optimizer.n_vecindarios", 4, 8)
        max_angle = trial.suggest_float("optimizer.max_angle", 70.0, 90.0)
        min_angle = trial.suggest_float("optimizer.min_angle", 2.0, 15.0)
        schedule_deg = np.geomspace(max_angle, min_angle, num=n_vecindarios).tolist()

        n_expl_low, n_expl_high, n_expl_step = search_space["n_exploration"]
        n_exploration = trial.suggest_int("optimizer.n_exploration", n_expl_low, n_expl_high, step=n_expl_step)
        test_multiplier = trial.suggest_int(
            "optimizer.test_multiplier", search_space["test_multiplier"][0], search_space["test_multiplier"][1]
        )
        n_test = max(1000, min(n_exploration * test_multiplier, 2500))
        n_evaluation = profile.n_evaluation
        n_bootstrap = profile.n_bootstrap

        b_samp_low, b_samp_high, b_samp_step = search_space["b_samples"]
        grid_t_low, grid_t_high, grid_t_step = search_space["grid_t_len"]

        config_payload = {
            "angular_schedules_deg": schedule_deg,
            "b_samples": trial.suggest_int("optimizer.b_samples", b_samp_low, b_samp_high, step=b_samp_step),
            "eta": trial.suggest_categorical("optimizer.eta", search_space["eta"]),
            "grid_t_len": trial.suggest_int("optimizer.grid_t_len", grid_t_low, grid_t_high, step=grid_t_step),
            "n_exploration": n_exploration,
            "test_multiplier": test_multiplier,
            "n_test": n_test,
            "n_evaluation": n_evaluation,
            "n_bootstrap": n_bootstrap,
        }

        tasks: list[dict[str, Any]] = []
        for case_index, case in enumerate(cases):
            d = int(case.artifact.x_train.shape[1])
            optimizer_kwargs = {
                "kappas": angular_schedule_to_kappas(schedule_deg, d),
                "b_samples": int(config_payload["b_samples"]),
                "n_exploration": int(config_payload["n_exploration"]),
                "n_test": int(config_payload["n_test"]),
                "n_evaluation": int(config_payload["n_evaluation"]),
                "n_bootstrap": int(config_payload["n_bootstrap"]),
                "eta": float(config_payload["eta"]),
                "grid_t": np.linspace(0.0, 1.0, int(config_payload["grid_t_len"])),
                "max_evals": None,
                "timeout_seconds": timeout_seconds,
                "trace_level": "none",
                "seed": seed + trial.number * 10_000 + case_index,
            }
            tasks.append(
                {
                    "case_index": case_index,
                    "optimizer_kwargs": optimizer_kwargs,
                }
            )

        if pool is not None:
            records = list(pool.map(_process_optuna_case, tasks))
        else:
            records = [_process_optuna_case(t) for t in tasks]

        for r in records:
            case_index = r["case_index"]
            case = cases[case_index]
            logger.info(
                "  [trial #%d] case %d/%d (%s) loss=%.6f time=%.4fs evals=%d stop=%s",
                trial.number,
                case_index + 1,
                len(cases),
                case.artifact.artifact_id,
                r["loss"],
                r["wall_time_seconds"],
                int(r["n_model_evals"]),
                r["stop_reason"],
            )

        objective_loss = float(np.mean([record["loss"] for record in records]))
        objective_time = float(np.mean([record["wall_time_seconds"] for record in records]))
        summary = {
            "n_cases": len(records),
            "mean_loss": objective_loss,
            "max_loss": float(np.max([record["loss"] for record in records])),
            "mean_time_seconds": objective_time,
            "mean_n_model_evals": float(np.mean([record["n_model_evals"] for record in records])),
            "timeout_seconds": timeout_seconds,
        }
        trial.set_user_attr("config_payload", config_payload)
        trial.set_user_attr("summary", summary)

        detailed_cases = []
        for r in records:
            case_index = r["case_index"]
            case = cases[case_index]
            detailed_cases.append(
                {
                    "case_index": case_index,
                    "artifact_id": case.artifact.artifact_id,
                    "query_id": case.query_id,
                    "robust_value": r["robust_value"],
                    "gold_value": r["gold_value"],
                    "loss": r["loss"],
                    "wall_time_seconds": r["wall_time_seconds"],
                    "n_model_evals": int(r["n_model_evals"]),
                    "stop_reason": r["stop_reason"],
                }
            )
        trial.set_user_attr("cases", detailed_cases)

        logger.info(
            "  [trial #%d] objective_loss: %.6f | objective_time: %.4fs | mean_n_model_evals: %.1f",
            trial.number,
            objective_loss,
            objective_time,
            summary["mean_n_model_evals"],
        )
        return objective_loss, objective_time

    return objective


def _search_space(profile: OptunaProfile) -> dict[str, Any]:
    """Retrieve the parameter search space definition from the profile.

    Args:
        profile (OptunaProfile): Optuna configuration profile.

    Returns:
        dict[str, Any]: Search space parameter boundaries and steps.
    """
    return profile.search_space


def _completed_trials(study: Any) -> list[Any]:
    """Filter and return completed trials with valid values.

    Args:
        study (Any): Optuna study to extract trials from.

    Returns:
        list[Any]: List of completed trials.
    """
    return [
        trial for trial in study.trials if trial.state is optuna.trial.TrialState.COMPLETE and trial.values is not None
    ]


def _summarize_study(study: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute Pareto frontiers and summary metrics for a study.

    Args:
        study (Any): Completed Optuna study.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]: Tuple containing:
            - Selected best balanced trial dictionary.
            - List of Pareto frontier trial dictionaries.
            - List of all trial summary rows.

    Raises:
        RuntimeError: If the study contains no completed trials.
    """
    completed = _completed_trials(study)
    if not completed:
        msg = "Optuna produced no completed trials."
        raise RuntimeError(msg)

    pareto = list(study.best_trials) or completed
    losses = [float(trial.values[0]) for trial in pareto]
    times = [float(math.log1p(trial.values[1])) for trial in pareto]
    scaled_losses = _minmax_scale(losses)
    scaled_times = _minmax_scale(times)

    pareto_trials: list[dict[str, Any]] = []
    for trial, scaled_loss, scaled_time in zip(pareto, scaled_losses, scaled_times, strict=True):
        pareto_trials.append(
            {
                "trial_number": trial.number,
                "precision_loss": float(trial.values[0]),
                "mean_time_seconds": float(trial.values[1]),
                "mean_n_model_evals": float(trial.user_attrs.get("summary", {}).get("mean_n_model_evals", 0.0)),
                "balance_score": max(scaled_loss, scaled_time),
                "euclidean_score": float(math.sqrt(scaled_loss**2 + scaled_time**2)),
                "config": dict(trial.user_attrs.get("config_payload", {})),
                "cases": list(trial.user_attrs.get("cases", [])),
            }
        )

    pareto_trials.sort(
        key=lambda item: (
            item["balance_score"],
            item["euclidean_score"],
            item["precision_loss"],
            item["mean_time_seconds"],
        )
    )
    selected = pareto_trials[0]

    trial_rows = [
        {
            "trial_number": trial.number,
            "is_pareto": any(pareto_trial.number == trial.number for pareto_trial in pareto),
            "precision_loss": float(trial.values[0]),
            "mean_time_seconds": float(trial.values[1]),
            "mean_n_model_evals": float(trial.user_attrs.get("summary", {}).get("mean_n_model_evals", 0.0)),
            "max_loss": float(trial.user_attrs.get("summary", {}).get("max_loss", 0.0)),
            "angular_schedules_deg": "|".join(
                str(value) for value in trial.user_attrs.get("config_payload", {}).get("angular_schedules_deg", [])
            ),
            "b_samples": int(trial.user_attrs.get("config_payload", {}).get("b_samples", 0)),
            "n_exploration": int(trial.user_attrs.get("config_payload", {}).get("n_exploration", 0)),
            "test_multiplier": int(trial.user_attrs.get("config_payload", {}).get("test_multiplier", 0)),
            "n_test": int(trial.user_attrs.get("config_payload", {}).get("n_test", 0)),
            "n_evaluation": int(trial.user_attrs.get("config_payload", {}).get("n_evaluation", 0)),
            "n_bootstrap": int(trial.user_attrs.get("config_payload", {}).get("n_bootstrap", 0)),
            "eta": float(trial.user_attrs.get("config_payload", {}).get("eta", 0.0)),
            "grid_t_len": int(trial.user_attrs.get("config_payload", {}).get("grid_t_len", 0)),
            "cases": list(trial.user_attrs.get("cases", [])),
        }
        for trial in completed
    ]
    trial_rows.sort(key=lambda row: (row["precision_loss"], row["mean_time_seconds"]))
    return selected, pareto_trials, trial_rows


def _selected_config_payload(
    bucket: DimensionBucket, selected_trial: dict[str, Any], profile: OptunaProfile
) -> dict[str, Any]:
    """Construct the YAML export payload for a selected configuration.

    Args:
        bucket (DimensionBucket): Dimensionality bucket name.
        selected_trial (dict[str, Any]): Dictionary of selected trial parameters and metrics.
        profile (OptunaProfile): Optuna configuration profile.

    Returns:
        dict[str, Any]: Structured configuration payload ready for YAML serialization.
    """
    timeout_seconds = profile.timeouts_by_bucket[bucket]
    config = selected_trial["config"]
    return {
        "dimension_bucket": bucket,
        "selection": {
            "trial_number": selected_trial["trial_number"],
            "precision_loss": selected_trial["precision_loss"],
            "mean_time_seconds": selected_trial["mean_time_seconds"],
            "mean_n_model_evals": selected_trial["mean_n_model_evals"],
            "balance_score": selected_trial["balance_score"],
        },
        "optimizer": {
            "angular_schedules_deg": list(config["angular_schedules_deg"]),
            "b_samples": int(config["b_samples"]),
            "eta": float(config["eta"]),
            "grid_t_len": int(config["grid_t_len"]),
            "n_exploration": int(config["n_exploration"]),
            "test_multiplier": int(config.get("test_multiplier", 1)),
            "n_test": int(config.get("n_test", config["n_exploration"])),
            "n_evaluation": int(config["n_evaluation"]),
            "n_bootstrap": int(config["n_bootstrap"]),
        },
        "execution": {
            "timeout_seconds": timeout_seconds,
        },
    }


def _minmax_scale(values: list[float]) -> list[float]:
    """Perform Min-Max scaling to map values to the [0, 1] range.

    Args:
        values (list[float]): List of numerical values to scale.

    Returns:
        list[float]: Scaled values in the [0, 1] range.
    """
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def main() -> None:
    """Run the hyperparameter tuning studies and persist the Pareto artifacts (CLI entry point)."""
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning.")
    parser.add_argument(
        "--bucket",
        type=str,
        default="all",
        choices=["low", "medium", "high", "all"],
        help="Dimension bucket to run tuning for.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel processes to use for evaluation (default: 1).",
    )
    args = parser.parse_args()

    seed = 20260530
    profile = default_optuna_profile()
    layout = experiment_layout()
    ensure_layout(layout)
    configure_logging(layout, "hyperparameter_tuning")

    bucket_arg = None if args.bucket == "all" else args.bucket

    logger.info(
        "Starting Optuna hyperparameter tuning studies (seed=%d, buckets=%s, workers=%d)",
        seed,
        args.bucket,
        args.workers,
    )
    payload = run_all_bucket_studies(profile, layout, seed=seed, bucket=bucket_arg, workers=args.workers)

    logger.info(
        "Completed %d Optuna studies.",
        len(payload["studies"]),
    )


if __name__ == "__main__":
    main()
