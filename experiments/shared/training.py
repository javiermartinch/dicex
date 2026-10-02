# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Train and persist the canonical models for every registered dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiments.shared.builders import make_classifier, make_regressor
from experiments.shared.datasets import DatasetLoadError, build_raw_dataset, default_dataset_registry
from experiments.shared.helpers import repo_relative
from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.specs import model_families_for_task

logger = get_logger("training")


def main() -> None:
    """Train the canonical models for every registered dataset (CLI entry point).

    Sweeps registered datasets, checks for pre-existing model checkpoints to avoid
    redundant fitting, standardizes feature subsets using standard scaling where
    needed, trains estimators (Linear, Logistic, RF, XGB, MLP), evaluates performance
    metrics, and persists model/scaler bundles to joblib files.
    """
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument(
        "--bucket",
        type=str,
        default="all",
        choices=["low", "medium", "high", "all"],
        help="Dimension bucket to process.",
    )
    args = parser.parse_args()

    configure_logging(script_name="training")

    output_dir = Path(__file__).resolve().parents[1] / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 42

    specs = default_dataset_registry()
    if args.bucket != "all":
        specs = [s for s in specs if s.dimension_bucket == args.bucket]

    logger.info("Found %d datasets in registry. Beginning training pipeline.", len(specs))

    all_metrics = []

    for spec in specs:
        logger.info("-" * 60)
        logger.info("Processing Dataset: %s (Task: %s)", spec.name, spec.task)

        families = model_families_for_task(spec.task)

        # Check if all models for this dataset already exist
        all_exist = all((output_dir / f"{spec.name}__{family}.joblib").exists() for family in families)
        if all_exist:
            logger.info("All models for dataset %s already exist. Loading cached metrics.", spec.name)
            skip_dataset = True
            temp_rows = []
            for family in families:
                filepath = output_dir / f"{spec.name}__{family}.joblib"
                try:
                    cached_artifact = joblib.load(filepath)
                    metrics_dict = cached_artifact.get("metrics", {})
                    row = {"dataset": spec.name, "task": spec.task, "model": family, **metrics_dict}
                    temp_rows.append(row)
                except (OSError, KeyError, ValueError, EOFError) as e:
                    logger.warning(
                        "Failed to load cached model %s: %s. Will reload dataset and retrain.", filepath.name, e
                    )
                    skip_dataset = False
                    break
            if skip_dataset:
                all_metrics.extend(temp_rows)
                continue

        try:
            raw_dataset = build_raw_dataset(spec, _seed=seed)
        except DatasetLoadError as e:
            logger.error("Failed to load dataset %s: %s", spec.name, e)
            continue

        x_all = np.asarray(raw_dataset.x, dtype=float)
        y_all = np.asarray(raw_dataset.y, dtype=float if spec.task == "regression" else int)

        # 80/20 Train/Test split
        x_train, x_test, y_train, y_test = train_test_split(
            x_all, y_all, test_size=0.20, random_state=seed, stratify=y_all if spec.task == "classification" else None
        )

        # Pre-compute scaled version for models that require it
        scaler_for_dataset = StandardScaler()
        x_train_scaled = scaler_for_dataset.fit_transform(x_train)
        x_test_scaled = scaler_for_dataset.transform(x_test)

        for family in families:
            filename = f"{spec.name}__{family}.joblib"
            filepath = output_dir / filename

            if filepath.exists():
                logger.info("  Model %s already exists. Loading cached metrics.", filename)
                try:
                    cached_artifact = joblib.load(filepath)
                    metrics_dict = cached_artifact.get("metrics", {})
                    row = {"dataset": spec.name, "task": spec.task, "model": family, **metrics_dict}
                    all_metrics.append(row)
                    continue
                except (OSError, KeyError, ValueError, EOFError) as e:
                    logger.warning("  Failed to load cached model %s: %s. Retraining...", filename, e)

            logger.info("  Training %s...", family)
            needs_scaling = family in ["linear", "logistic", "mlp"]
            if needs_scaling:
                x_train_model = x_train_scaled
                x_test_model = x_test_scaled
                active_scaler = scaler_for_dataset
            else:
                x_train_model = x_train
                x_test_model = x_test
                active_scaler = None

            if spec.task == "regression":
                model = make_regressor(family, seed=seed)
            else:
                model = make_classifier(family, seed=seed)

            model.fit(x_train_model, y_train)

            train_score = float(model.score(x_train_model, y_train))
            test_score = float(model.score(x_test_model, y_test))

            y_pred_train = model.predict(x_train_model)
            y_pred_test = model.predict(x_test_model)

            metrics_dict = {
                "train_score": train_score,
                "test_score": test_score,
            }

            if spec.task == "regression":
                metrics_dict.update(
                    {
                        "train_mse": float(mean_squared_error(y_train, y_pred_train)),
                        "test_mse": float(mean_squared_error(y_test, y_pred_test)),
                        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
                        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
                    }
                )
            else:
                metrics_dict.update(
                    {
                        "train_f1_macro": float(f1_score(y_train, y_pred_train, average="macro")),
                        "test_f1_macro": float(f1_score(y_test, y_pred_test, average="macro")),
                        "train_precision_macro": float(
                            precision_score(y_train, y_pred_train, average="macro", zero_division=0)
                        ),
                        "test_precision_macro": float(
                            precision_score(y_test, y_pred_test, average="macro", zero_division=0)
                        ),
                        "train_recall_macro": float(recall_score(y_train, y_pred_train, average="macro")),
                        "test_recall_macro": float(recall_score(y_test, y_pred_test, average="macro")),
                    }
                )

            logger.info("    -> Train Score: %.4f | Test Score: %.4f", train_score, test_score)

            # Persist model and scaler
            artifact = {
                "dataset_name": spec.name,
                "model_family": family,
                "task": spec.task,
                "model": model,
                "scaler": active_scaler,  # None for tree-based models
                "feature_names": raw_dataset.feature_names,
                "split_seed": seed,
                "metrics": metrics_dict,
            }

            row = {"dataset": spec.name, "task": spec.task, "model": family, **metrics_dict}
            all_metrics.append(row)

            filename = f"{spec.name}__{family}.joblib"
            filepath = output_dir / filename
            joblib.dump(artifact, filepath)

    logger.info("    -> Saved artifact to %s", repo_relative(filepath))

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        metrics_csv_path = output_dir / "metrics.csv"
        df_metrics.to_csv(metrics_csv_path, index=False)
        logger.info("Saved consolidated metrics to %s", repo_relative(metrics_csv_path))

    logger.info("=" * 60)
    logger.info("All canonical models trained and persisted successfully.")


if __name__ == "__main__":
    main()
