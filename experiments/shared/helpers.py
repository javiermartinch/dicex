# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Miscellaneous helpers: repo-relative paths, angular schedules, and prediction functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dicex.optim.vmf import kappa_from_angular_radius

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_relative(path: str | Path) -> str:
    """Convert an absolute path to a repo-relative path for privacy-safe logging.

    Args:
        path (str | Path): Path to convert.

    Returns:
        str: String containing the repo-relative path, or the original path as a string
            if conversion fails.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def angular_schedule_to_kappas(schedule_deg: list[float], d: int) -> list[float]:
    """Convert an angular schedule expressed in degrees into vMF concentration parameter kappas.

    Args:
        schedule_deg (list[float]): Angular radius values in degrees.
        d (int): Dimension of the space.

    Returns:
        list[float]: List of kappa concentration parameters.
    """
    return [float(kappa_from_angular_radius(np.deg2rad(deg), d)) for deg in schedule_deg]


def load_optimal_config(
    bucket: str = "low",
    *,
    results_root: Path | None = None,
) -> dict[str, Any]:
    """Load the recommended hyperparameter config for a dimension bucket.

    Args:
        bucket (str): Name of the dimension bucket ('low', 'medium', or 'high').
        results_root (Path | None): Custom root path of the tuning results. If None, uses the
            default package path.

    Returns:
        dict[str, Any]: Dictionary of recommended DiCEx optimizer hyperparameters.
    """
    if results_root is None:
        results_root = REPO_ROOT / "experiments" / "hyperparameter_tuning" / "results" / "selected_configs"
    config_path = results_root / f"{bucket}_recommended.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_predict_fn(artifact: Any) -> Any:
    """Return a unified prediction function returning a 1D array of values.

    For regression models, it returns a callable evaluating ``predict(X)``.
    For classification models, it returns a callable evaluating
    ``predict_proba(X)[:, target_class]``. Handles scaling transparently if a
    scaler is attached to the artifact.

    Args:
        artifact (Any): The TrainedModelArtifact or similar bundle holding the model,
            task type, target class, and optional scaler.

    Returns:
        Any: A callable wrapping the prediction pipeline.
    """
    model = _get_scaled_model_wrapper(artifact) if getattr(artifact, "scaler", None) is not None else artifact.model

    if artifact.task == "regression":
        return lambda x: np.asarray(model.predict(x), dtype=float)
    target = getattr(artifact, "target_class", None)
    if target is None:
        target = 1
    return lambda x: np.asarray(model.predict_proba(x)[:, target], dtype=float)


def _get_scaled_model_wrapper(artifact: Any) -> Any:
    """Return a ScaledModelWrapper instance for an artifact using lazy import.

    Args:
        artifact (Any): Trained model artifact bundle containing model and scaler.

    Returns:
        Any: ScaledModelWrapper wrapping the model and scaler.
    """
    from experiments.shared.evaluation import ScaledModelWrapper  # noqa: PLC0415

    return ScaledModelWrapper(artifact.model, artifact.scaler)
