# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Load trained model artifacts from disk and rebuild their train/test splits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from experiments.shared.bundles import TrainedModelArtifact
from experiments.shared.datasets import build_raw_dataset, default_dataset_registry
from experiments.shared.log_config import get_logger

if TYPE_CHECKING:
    from experiments.shared.persistence import RunLayout

logger = get_logger("loader")


def load_trained_artifacts(
    layout: RunLayout, bucket: str | None = None, model_families: list[str] | None = None
) -> list[TrainedModelArtifact]:
    """Load and reconstruct TrainedModelArtifact objects from the file system.

    Sweeps the central model directories for scikit-learn/joblib models and
    reconstructs standard TrainedModelArtifact objects by re-splitting the
    source datasets matching the metadata.

    Args:
        layout (RunLayout): Folder layout configuration mapping central folders.
        bucket (str | None): Optional size filter ('low', 'medium', or 'high') to restrict loaded
            datasets.
        model_families (list[str] | None): Optional list of model families to load (e.g. ['linear',
            'xgboost']).

    Returns:
        list[TrainedModelArtifact]: A list of loaded and reconstructed TrainedModelArtifact objects.
    """
    logger.info("Sweeping experiments/models/ for pretrained models...")

    models_dir = layout.models_root

    specs = {s.name: s for s in default_dataset_registry()}
    artifacts: list[TrainedModelArtifact] = []

    # Load scikit-learn compatible models (.joblib)
    for joblib_file in models_dir.glob("*.joblib"):
        name_parts = joblib_file.stem.split("__")
        if len(name_parts) == 2:
            file_family = name_parts[1]
            if model_families is not None and file_family not in model_families:
                continue

        logger.info("Loading %s", joblib_file.name)
        data = joblib.load(joblib_file)

        ds_name = data["dataset_name"]
        spec = specs.get(ds_name)
        if spec is None:
            continue

        if bucket is not None and spec.dimension_bucket != bucket:
            continue

        seed = data["split_seed"]
        raw_dataset = build_raw_dataset(spec, _seed=seed)
        x_all = np.asarray(raw_dataset.x, dtype=float)
        y_all = np.asarray(raw_dataset.y, dtype=float if spec.task == "regression" else int)

        x_train, x_test, y_train, y_test = train_test_split(
            x_all, y_all, test_size=0.20, random_state=seed, stratify=y_all if spec.task == "classification" else None
        )

        scaler = data.get("scaler")

        artifacts.append(
            TrainedModelArtifact(
                dataset_name=ds_name,
                task=spec.task,
                model_family=data["model_family"],
                dimension_bucket=spec.dimension_bucket,
                feature_names=tuple(data["feature_names"]),
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                model=data["model"],
                imputer=None,
                scaler=scaler,
                target_class=1 if spec.task == "classification" else None,
                query_points=(),
                dataset_summary={"n_samples_raw": len(x_all), "metrics": data.get("metrics")},
                model_summary={},
            )
        )

    logger.info("Reconstructed %d artifacts.", len(artifacts))
    return artifacts
