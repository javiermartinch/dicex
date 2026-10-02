# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for the core functional logic."""

import numpy as np
import pytest

from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.core.functional import (
    directional_improvement,
    directional_improvement_batch,
    robust_directional_value,
)
from dicex.core.scoring import ModelScorer
from dicex.distributions.gaussian import GaussianPerturbation


class MockModel:
    """Mock model for testing."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Linear model: sum of features."""
        return np.sum(x, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Dummy classification: probability of class 1 proportional to sum."""
        probs = np.zeros((len(x), 2))
        s = np.sum(x, axis=1)
        # Sigmoid-like for testing
        p = 1 / (1 + np.exp(-s))
        probs[:, 1] = p
        probs[:, 0] = 1 - p
        return probs


def test_directional_improvement_regression() -> None:
    """Test directional improvement with a linear regression-like model."""
    model = MockModel()
    x0 = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    noise = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])

    # No env noise: Z_i = M(x0 + noise_i*c) - M(x0)
    # Prediction at perturbed points [1, 0], [2, 0], [-1, 0] are 1, 2, -1.
    # Baseline prediction at [0, 0] is 0.
    # Expected Z -> [1.0, 2.0, -1.0]

    z = directional_improvement(model, x0, c, noise)
    assert np.allclose(z, [1.0, 2.0, -1.0])


def test_directional_improvement_with_env_noise() -> None:
    """Test directional improvement with additive environmental noise."""
    model = MockModel()
    x0 = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    noise = np.array([[1.0, 0.0], [2.0, 0.0]])
    # env_noise adds [0.5, 0.5] to each point
    env_noise = np.array([[0.5, 0.5], [0.0, -1.0]])

    # x0 + T_1*c + ε_1 = [0,0] + [1,0] + [0.5,0.5] = [1.5, 0.5] -> sum = 2.0
    # x0 + T_2*c + ε_2 = [0,0] + [2,0] + [0.0,-1.0] = [2.0, -1.0] -> sum = 1.0
    # The reference value is 0.0
    # Z -> [2.0, 1.0]

    z = directional_improvement(model, x0, c, noise, env_noise)
    assert np.allclose(z, [2.0, 1.0])


def test_directional_improvement_classification() -> None:
    """Test directional improvement with a classification model."""
    model = MockModel()
    x0 = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    noise = np.array([[0.0, 0.0], [10.0, 0.0]])  # 10 is high, p will be near 1

    # p(0) = 0.5
    # p(10) approx 1.0
    # Z -> [0, approx 0.5]

    z = directional_improvement(model, x0, c, noise, target_class=1)
    assert np.isclose(z[0], 0.0)
    assert z[1] > 0.4


def test_robust_directional_value() -> None:
    """Test robust directional value calculation."""
    model = MockModel()
    x0 = np.array([0.0])
    c = np.array([1.0])
    perturbation = GaussianPerturbation(mu=0.0, sigma=1.0, seed=42)
    alpha = 0.5

    # CVaR of Z(c) where Z approx N(0, 1)
    # For alpha=0.5, CVaR is the mean of the lower half of the normal distribution.
    val = robust_directional_value(model, x0, c, perturbation, alpha, n_samples=1000)

    # Standard normal mean is 0, so mean of lower half must be negative.
    assert val < 0.0


def test_robust_directional_value_with_rng_matches_manual_sampling() -> None:
    """Explicit RNG should reproduce the direct final-evaluation sampling path."""
    model = MockModel()
    x0 = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    perturbation = GaussianPerturbation(mu=0.3, sigma=0.2, env_cov=np.array([0.01, 0.04]))
    alpha = 0.5
    n_samples = 16
    seed = 123

    manual_rng = np.random.default_rng(seed)
    noise = perturbation.sample(n_samples, rng=manual_rng)
    env_noise = perturbation.sample_env(n_samples, len(x0), rng=manual_rng)
    z_samples = directional_improvement(model, x0, c, noise, env_noise)
    expected = lower_tail_cvar_empirical(z_samples, alpha)

    observed = robust_directional_value(
        model,
        x0,
        c,
        perturbation,
        alpha,
        n_samples=n_samples,
        rng=np.random.default_rng(seed),
    )

    assert np.isclose(observed, expected)


def test_directional_improvement_batch_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test batch improvement uses Python fallback when Rust is disabled."""
    import dicex.core.functional  # noqa: PLC0415

    monkeypatch.setattr(dicex.core.functional, "HAS_RUST_BATCH", False)

    model = MockModel()
    scorer = ModelScorer(model)
    x0 = np.array([0.0, 0.0])
    directions = np.array([[1.0, 0.0], [0.0, 1.0]])
    noise = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
    env_noise = np.zeros((3, 2))

    result = directional_improvement_batch(scorer, x0, directions, noise, env_noise)
    assert result.shape == (2, 3)
    assert np.allclose(result[0], [1.0, 2.0, -1.0])
    assert np.allclose(result[1], [0.0, 0.0, 0.0])
