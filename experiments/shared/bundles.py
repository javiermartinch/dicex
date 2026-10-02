# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Container for a trained dataset/model artifact used across the experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from experiments.shared.query_points import QueryPoint

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class TrainedModelArtifact:
    """Trained model bundle representing a complete dataset/model artifact.

    Contains training and test splits, scaler/imputer, and summaries of the trained
    predictor model for use by downstream evaluation and explanation tasks.

    Attributes:
        dataset_name (str): Name of the source dataset.
        task (str): Problem task type ('regression' or 'classification').
        model_family (str): Underlying model family (e.g. 'linear', 'xgboost').
        dimension_bucket (str): Dataset size category ('low', 'medium', 'high').
        feature_names (tuple[str, ...]): Names of features in the input data.
        x_train (FloatArray): Training feature matrix.
        y_train (FloatArray | IntArray): Training target array.
        x_test (FloatArray | None): Test feature matrix.
        y_test (FloatArray | IntArray | None): Test target array.
        model (Any): Trained predictor model.
        imputer (SimpleImputer | None): Preprocessing imputer applied to features, if any.
        scaler (StandardScaler | None): Preprocessing scaler applied to features, if any.
        target_class (int | None): Classification target class index, if applicable.
        query_points (tuple[QueryPoint, ...]): Selected evaluation query points.
        dataset_summary (dict[str, Any]): Meta-information about the dataset structure.
        model_summary (dict[str, Any]): Performance metrics and model hyperparameters.
    """

    dataset_name: str
    task: str
    model_family: str
    dimension_bucket: str
    feature_names: tuple[str, ...]
    x_train: FloatArray
    y_train: FloatArray | IntArray
    x_test: FloatArray | None
    y_test: FloatArray | IntArray | None
    model: Any
    imputer: SimpleImputer | None
    scaler: StandardScaler | None
    target_class: int | None
    query_points: tuple[QueryPoint, ...]
    dataset_summary: dict[str, Any]
    model_summary: dict[str, Any]

    @property
    def artifact_id(self) -> str:
        """Return the unique string key identifying this trained model artifact.

        Returns:
            str: Identifier string in the format 'dataset_name__model_family'.
        """
        return f"{self.dataset_name}__{self.model_family}"
