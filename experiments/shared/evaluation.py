# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Helpers to run DiCEx on trained artifacts and to compute closed-form references."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from dicex.analytical.linear import linear_directional_functional, linear_optimal_direction
from dicex.analytical.logistic import logistic_directional_functional, logistic_optimal_direction
from dicex.distributions.gaussian import GaussianPerturbation
from dicex.explainer import Dicex
from dicex.preprocessing.scaling import DicexScaler

if TYPE_CHECKING:
    from experiments.shared.bundles import TrainedModelArtifact
    from experiments.shared.specs import NoiseRiskConfig

FloatArray = NDArray[np.float64]
EPSILON = 1e-7


def make_perturbation(noise_config: NoiseRiskConfig, d: int) -> GaussianPerturbation:
    """Build the Gaussian perturbation used across the evaluation pipeline.

    Args:
        noise_config (NoiseRiskConfig): Noise parameters specifying mu, sigma, env_std.
        d (int): Dimension of the input space.

    Returns:
        GaussianPerturbation: A GaussianPerturbation distribution instance.
    """
    env_cov = np.full(d, noise_config.env_std**2, dtype=float)
    return GaussianPerturbation(mu=noise_config.mu, sigma=noise_config.sigma, env_cov=env_cov)


class ScaledModelWrapper:
    """Model adapter that scales input features before passing them to a model.

    Wraps a model together with its associated scaler for pipeline evaluation.
    """

    def __init__(self, model: Any, scaler: Any) -> None:
        """Initialize the wrapper.

        Args:
            model (Any): Estimator model.
            scaler (Any): Scaler transformer (e.g. StandardScaler).
        """
        self.model = model
        self.scaler = scaler

    def predict(self, x: FloatArray) -> FloatArray:
        """Scale and predict continuous targets.

        Args:
            x (FloatArray): Input feature array.

        Returns:
            FloatArray: Model predictions.
        """
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled)

    def predict_proba(self, x: FloatArray) -> FloatArray:
        """Scale and predict class probabilities.

        Args:
            x (FloatArray): Input feature array.

        Returns:
            FloatArray: Predicted probability matrix.
        """
        x_scaled = self.scaler.transform(x)
        return self.model.predict_proba(x_scaled)


def run_artifact_explanation(
    artifact: TrainedModelArtifact,
    *,
    x0: FloatArray,
    noise_config: NoiseRiskConfig,
    optimizer_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run DiCEx on one trained artifact and return a flat result payload.

    Args:
        artifact (TrainedModelArtifact): Reconstructed trained-model bundle.
        x0 (FloatArray): Baseline query point.
        noise_config (NoiseRiskConfig): Perturbation noise parameters.
        optimizer_kwargs (dict[str, Any]): Custom solver configuration dictionary.

    Returns:
        dict[str, Any]: Dictionary with keys 'direction', 'robust_value', and 'metadata'.
    """
    perturbation = make_perturbation(noise_config, len(x0))
    model_to_explain = (
        ScaledModelWrapper(artifact.model, artifact.scaler) if artifact.scaler is not None else artifact.model
    )
    if artifact.scaler is not None:
        scaler_arg = DicexScaler()
        scaler_arg.mean_ = np.asarray(artifact.scaler.mean_, dtype=float)
        scaler_arg.std_ = np.asarray(artifact.scaler.scale_, dtype=float)
    else:
        scaler_arg = "auto"

    explainer = Dicex(
        model=model_to_explain,
        task=artifact.task,
        target_class=artifact.target_class,
        perturbation=perturbation,
        alpha=noise_config.alpha,
        scaler=scaler_arg,
        verbose="none",
        **optimizer_kwargs,
    )
    if scaler_arg == "auto":
        explainer.fit(artifact.x_train)

    result = explainer.explain(np.asarray(x0, dtype=float))
    return {
        "direction": np.asarray(result.direction, dtype=float),
        "robust_value": float(result.robust_value),
        "metadata": dict(result.metadata),
    }


def analytic_reference_result(
    artifact: TrainedModelArtifact,
    *,
    x0: FloatArray,
    noise_config: NoiseRiskConfig,
) -> dict[str, Any] | None:
    """Return an exact closed-form analytical reference when the model supports it.

    Args:
        artifact (TrainedModelArtifact): Trained model artifact bundle.
        x0 (FloatArray): Baseline query point.
        noise_config (NoiseRiskConfig): Perturbation specifications.

    Returns:
        dict[str, Any] | None: Dictionary with analytical direction and value, or None if the model is not
            a linear/logistic oracle.
    """
    perturbation = make_perturbation(noise_config, len(x0))
    if artifact.model_family == "linear":
        coef = artifact.model.coef_
        if coef.ndim > 1:
            # Multi-output regression has no closed-form reference; the analytic
            # solver supports single-target regression only.
            return None
        beta_raw = np.asarray(coef, dtype=float).reshape(-1)
        if artifact.scaler is not None:
            beta_norm = beta_raw
        else:
            std = np.std(artifact.x_train, axis=0)
            std[std < 1e-8] = 1.0
            beta_norm = beta_raw * std
        direction = linear_optimal_direction(beta_norm, perturbation, noise_config.alpha)
        robust_value = linear_directional_functional(beta_norm, perturbation, noise_config.alpha, direction)
        return {
            "direction": np.asarray(direction, dtype=float),
            "robust_value": float(robust_value),
            "metadata": {"n_model_evals": 0, "stop_reason": "analytic", "wall_time_seconds": 0.0},
            "reference_kind": "analytic",
        }
    if artifact.model_family == "logistic":
        coef = artifact.model.coef_
        intercept = artifact.model.intercept_
        if coef.shape[0] > 1:
            # Multiclass logistic regression: select the class with highest prediction variance on training set
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

        direction = logistic_optimal_direction(beta_norm, intercept_norm, x0_eval, perturbation, noise_config.alpha)
        robust_value = logistic_directional_functional(
            beta_norm,
            intercept_norm,
            x0_eval,
            perturbation,
            noise_config.alpha,
            direction,
        )
        return {
            "direction": np.asarray(direction, dtype=float),
            "robust_value": float(robust_value),
            "metadata": {"n_model_evals": 0, "stop_reason": "analytic", "wall_time_seconds": 0.0},
            "reference_kind": "analytic",
        }
    return None
