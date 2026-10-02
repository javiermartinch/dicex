# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Tests for the BasePerturbation ABC."""

from typing import Any

import numpy as np
import pytest

from dicex.distributions.base import BasePerturbation
from dicex.exceptions import InvalidParameterError


def test_base_perturbation_cannot_be_instantiated() -> None:
    """Test that the ABC cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BasePerturbation()  # type: ignore[abstract]


def test_base_perturbation_subclass_instantiation() -> None:
    """Test that a subclass implementing all methods can be instantiated."""

    class MockPerturbation(BasePerturbation):
        def sample(self, n: int, d: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
            del rng
            return np.zeros((n, d or 1))

        def sample_env(self, n: int, d: int, rng: np.random.Generator | None = None) -> np.ndarray:
            del rng
            return np.zeros((n, d))

        def projected_env_mean(self, beta: np.ndarray) -> float:
            return 0.0

        def projected_env_var(self, beta: np.ndarray) -> float:
            return 0.0

        @property
        def params(self) -> dict[str, Any]:
            return {"foo": "bar"}

    mock = MockPerturbation()
    assert mock.params == {"foo": "bar"}
    assert mock.sample_env(5, 3).shape == (5, 3)
    assert mock.projected_env_mean(np.ones(3)) == 0.0
    assert mock.projected_env_var(np.ones(3)) == 0.0


def test_base_sample_env_rng_none() -> None:
    """Test that sample_env works when rng is None (covers BasePerturbation path)."""

    class SimplePerturbation(BasePerturbation):
        def sample(self, n: int, d: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
            del rng
            return np.zeros((n, d or 1))

        @property
        def params(self) -> dict[str, Any]:
            return {}

    dist = SimplePerturbation()
    samples = dist.sample_env(5, 3, rng=None)
    assert samples.shape == (5, 3)
    assert np.allclose(samples, 0.0)


def test_base_sample_env_env_mean_mismatch() -> None:
    """Test that mismatched env_mean length raises InvalidParameterError."""

    class SimplePerturbation(BasePerturbation):
        def sample(self, n: int, d: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
            del rng
            return np.zeros((n, d or 1))

        @property
        def params(self) -> dict[str, Any]:
            return {}

    dist = SimplePerturbation()
    dist.env_mean = np.array([1.0, 2.0])
    with pytest.raises(InvalidParameterError, match="env_mean has length 2, but d=3"):
        dist.sample_env(5, 3)


def test_base_projected_env_var_1d_cov() -> None:
    """Test projected_env_var with a 1D (diagonal) covariance."""

    class SimplePerturbation(BasePerturbation):
        def sample(self, n: int, d: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
            del rng
            return np.zeros((n, d or 1))

        @property
        def params(self) -> dict[str, Any]:
            return {}

    dist = SimplePerturbation()
    dist.env_cov = np.array([1.0, 2.0, 3.0])
    beta = np.array([1.0, 1.0, 1.0])
    # Expected: β^T diag(Σ) β = 1²·1 + 1²·2 + 1²·3 = 6
    assert np.isclose(dist.projected_env_var(beta), 6.0)
