# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
# pyright: reportUnknownVariableType=false, reportPossiblyUnboundVariable=false, reportConstantRedefinition=false
"""Core functional computations for the robust directional functional.

Implements the black-box Monte Carlo evaluation of:
    Z(c) = M(x0 + T c + epsilon) - M(x0)
    Delta_alpha(c) = LCVaR_alpha(Z(c))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.core.geometry import apply_directional_noise
from dicex.core.scoring import ModelScorer
from dicex.logging import get_logger

if TYPE_CHECKING:
    from dicex.core.model_protocols import ClassifierModel, RegressorModel
    from dicex.distributions.base import BasePerturbation

_logger = get_logger(__name__)

try:
    from dicex._rust_ext import generate_batched_points_fast

    HAS_RUST_BATCH = True
except ImportError:  # pragma: no cover
    HAS_RUST_BATCH = False

_NOISE_2D = 2


def _ensure_scorer(
    model: ModelScorer | RegressorModel | ClassifierModel, target_class: int | None = None
) -> ModelScorer:
    """Wrap a model in a `ModelScorer` if it is not one already.

    Args:
        model (ModelScorer | RegressorModel | ClassifierModel): The model to wrap.
        target_class (int | None): The target class index for classifiers.

    Returns:
        ModelScorer: The wrapped instance.
    """
    if isinstance(model, ModelScorer):
        return model
    return ModelScorer(model, target_class=target_class)


def directional_improvement(
    model: ModelScorer | RegressorModel | ClassifierModel,
    x0: np.ndarray,
    c: np.ndarray,
    noise_samples: np.ndarray,
    env_noise_samples: np.ndarray | None = None,
    target_class: int | None = None,
) -> np.ndarray:
    """Compute the directional improvement samples Z_i(c).

    Z_i(c) = M(x0 + T_i ⊙ c + ε_i) - M(x0)

    Args:
        model (ModelScorer | RegressorModel | ClassifierModel): A :class:`ModelScorer` or a raw model
            (will be wrapped).
        x0 (np.ndarray): Baseline point (1D array).
        c (np.ndarray): Direction vector (1D array).
        noise_samples (np.ndarray): Directional noise samples T_i (2D array, shape (n, d)).
        env_noise_samples (np.ndarray | None): Additive noise samples \u03b5_i (2D array,
            shape (n, d)). If None, no additive noise is applied.
        target_class (int | None): Optional index for classification models. Ignored when
            *model* is already a :class:`ModelScorer` (uses its target_class).

    Returns:
        np.ndarray: 1D array of improvement samples.
    """
    scorer = _ensure_scorer(model, target_class)

    # 1. Compute baseline prediction
    y0 = scorer.score_baseline(x0)

    # 2. Compute perturbed points: x0 + T_i ⊙ c + ε_i
    perturbed_x = apply_directional_noise(x0, c, noise_samples, env_noise_samples)

    # 3. Compute perturbed predictions
    y_perturbed = scorer.score(perturbed_x)

    return np.asarray(y_perturbed - y0)


def directional_improvement_batch(
    scorer: ModelScorer,
    x0: np.ndarray,
    directions: np.ndarray,
    noise_samples: np.ndarray,
    env_noise_samples: np.ndarray,
) -> np.ndarray:
    """Compute improvements for multiple directions in one batched model call.

    Args:
        scorer (ModelScorer): A :class:`ModelScorer` (with optional tracking).
        x0 (np.ndarray): Baseline point (1D array of shape (d,)).
        directions (np.ndarray): Candidate directions, shape (n_dirs, d).
        noise_samples (np.ndarray): Directional noise samples, shape (n, d).
        env_noise_samples (np.ndarray): Additive noise samples, shape (n, d).

    Returns:
        np.ndarray: 2D array of shape (n_dirs, n) with improvement values per direction.
    """
    y0 = scorer.score_baseline(x0)

    n_samples = env_noise_samples.shape[0]
    n_directions = directions.shape[0]
    d = len(x0)

    if HAS_RUST_BATCH:
        points = generate_batched_points_fast(
            np.ascontiguousarray(x0, dtype=np.float64),
            np.ascontiguousarray(directions, dtype=np.float64),
            np.ascontiguousarray(noise_samples, dtype=np.float64),
            np.ascontiguousarray(env_noise_samples, dtype=np.float64),
        )
    else:
        _logger.debug("Using pure-Python batch path (Rust extension unavailable)")
        if noise_samples.ndim == _NOISE_2D and noise_samples.shape[1] == d:
            points_3d = x0 + noise_samples[:, np.newaxis, :] * directions[np.newaxis, :, :]
        else:  # pragma: no cover
            points_3d = x0 + noise_samples[:, np.newaxis, np.newaxis] * directions[np.newaxis, :, :]

        points_3d = points_3d + env_noise_samples[:, np.newaxis, :]
        points = points_3d.reshape(n_samples * n_directions, d)

    scores = scorer.score(points)  # pyright: ignore[reportUnknownArgumentType]
    return scores.reshape(n_samples, n_directions).T - y0


def robust_directional_value(  # noqa: PLR0913
    model: ModelScorer | RegressorModel | ClassifierModel,
    x0: np.ndarray,
    c: np.ndarray,
    perturbation: BasePerturbation,
    alpha: float,
    n_samples: int,
    target_class: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Compute the robust directional value CVaR_alpha(Z(c)).

    Args:
        model (ModelScorer | RegressorModel | ClassifierModel): Prediction model or scorer.
        x0 (np.ndarray): Baseline point.
        c (np.ndarray): Direction vector.
        perturbation (BasePerturbation): Distribution to sample (T, \u03b5) from.
        alpha (float): Risk level of the lower-tail CVaR.
        n_samples (int): Number of samples for empirical CVaR.
        target_class (int | None): Optional target class for classification.
        rng (np.random.Generator | None): Optional random generator for deterministic evaluation.

    Returns:
        float: The robust improvement value.
    """
    noise = perturbation.sample(n_samples, d=len(x0), rng=rng)
    d = len(x0)
    env_noise = perturbation.sample_env(n_samples, d, rng=rng)
    z_samples = directional_improvement(model, x0, c, noise, env_noise, target_class)
    return lower_tail_cvar_empirical(z_samples, alpha)
