# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Selection of query points at fixed percentiles of the test-set predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from experiments.shared.helpers import get_predict_fn

FloatArray = NDArray[np.float64]

DEFAULT_QUERY_PERCENTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass(frozen=True)
class QueryPoint:
    """One evaluation point selected from a trained dataset."""

    query_id: str
    percentile: float
    point_index: int
    point_value: float
    x0: FloatArray


def _percentile_label(percentile: float) -> str:
    """Format a percentile float as a standardized ID label.

    Args:
        percentile (float): Fraction value (e.g. 0.50).

    Returns:
        str: String label (e.g. 'p50').
    """
    return f"p{round(percentile * 100):02d}"


def unique_nearest_indices(values: FloatArray, percentiles: tuple[float, ...]) -> list[int]:
    """Select unique sample indices in the array nearest to the requested percentiles.

    Args:
        values (FloatArray): 1D array of predicted target values.
        percentiles (tuple[float, ...]): Tuple of percentile levels (e.g. 0.10, 0.50, 0.90).

    Returns:
        list[int]: A list of distinct integer indices matching the percentiles.
    """
    targets = np.quantile(values, percentiles)
    selected: list[int] = []
    used: set[int] = set()
    for target in targets:
        ranking = np.argsort(np.abs(values - target))
        for candidate in ranking:
            index = int(candidate)
            if index not in used:
                used.add(index)
                selected.append(index)
                break
    return selected


def select_query_points(
    x_data: FloatArray,
    point_values: FloatArray,
    percentiles: tuple[float, ...] = DEFAULT_QUERY_PERCENTILES,
) -> tuple[QueryPoint, ...]:
    """Build the fixed query point set from features and predictions.

    Args:
        x_data (FloatArray): Feature matrix.
        point_values (FloatArray): Model predictions for x_data.
        percentiles (tuple[float, ...]): Tuple of target percentile levels.

    Returns:
        tuple[QueryPoint, ...]: A tuple of selected QueryPoint objects.
    """
    indices = unique_nearest_indices(point_values, percentiles)
    query_points: list[QueryPoint] = []
    for percentile, index in zip(percentiles, indices, strict=True):
        query_points.append(
            QueryPoint(
                query_id=_percentile_label(percentile),
                percentile=float(percentile),
                point_index=index,
                point_value=float(point_values[index]),
                x0=np.asarray(x_data[index], dtype=float),
            )
        )
    return tuple(query_points)


def compute_query_points_on_test(
    artifacts: list[Any],
    percentiles: tuple[float, ...] = DEFAULT_QUERY_PERCENTILES,
) -> None:
    """Compute query points for each artifact independently using its test predictions.

    Args:
        artifacts (list[Any]): List of TrainedModelArtifact objects.
        percentiles (tuple[float, ...]): Tuple of target percentile levels.

    Raises:
        ValueError: If an artifact is missing the x_test array.
    """
    for artifact in artifacts:
        if not hasattr(artifact, "x_test") or artifact.x_test is None:
            msg = f"Artifact {artifact.artifact_id} is missing x_test. Make sure it is loaded correctly."
            raise ValueError(msg)

        predict_fn = get_predict_fn(artifact)
        point_values = predict_fn(artifact.x_test)
        qps = select_query_points(artifact.x_test, point_values, percentiles)
        artifact.query_points = qps
