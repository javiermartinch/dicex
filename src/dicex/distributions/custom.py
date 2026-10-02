# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Custom perturbation distribution.

Allows wrapping arbitrary callables for the pair (T, ε), with T valued in R^d.
"""

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from dicex.distributions.base import BasePerturbation
from dicex.exceptions import InvalidParameterError


class CustomPerturbation(BasePerturbation):
    """Perturbation distribution defined by user-provided sampling callables."""

    def __init__(
        self,
        sample_fn: Callable[[int], np.ndarray] | Callable[[int, int], np.ndarray],
        sample_env_fn: Callable[[int, int], np.ndarray] | None = None,
        env_mean: np.ndarray | None = None,
        env_cov: np.ndarray | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with user-provided callables.

        Args:
            sample_fn (Callable): Callable that returns an array of directional
                samples of shape (n, d).
            sample_env_fn (Callable | None): Optional callable
                (n, d) -> (n, d) array for ε samples.
            env_mean (np.ndarray | None): Mean vector m_ε of the additive noise (d,).
            env_cov (np.ndarray | None): Covariance matrix Σ_ε of the
                additive noise (d, d).
            params (dict[str, Any] | None): Optional dict of parameters
                describing the distribution.
        """
        self._sample_fn = sample_fn
        self._sample_env_fn = sample_env_fn
        self.env_mean = np.asarray(env_mean, dtype=float) if env_mean is not None else None
        self.env_cov = np.asarray(env_cov, dtype=float) if env_cov is not None else None
        self._params = params or {}

    def sample(self, n: int, d: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n directional perturbation vectors T_i.

        Args:
            n (int): Number of samples.
            d (int | None): Optional feature-space dimension.
            rng (np.random.Generator | None): Optional random number generator.

        Returns:
            np.ndarray: Array of shape (n,) when d is None or the callable
                ignores d, or (n, d) otherwise.
        """
        del rng
        if d is None:
            return np.asarray(cast("Any", self._sample_fn)(n), dtype=float)
        try:
            return np.asarray(cast("Any", self._sample_fn)(n, d), dtype=float)
        except TypeError:
            return np.asarray(cast("Any", self._sample_fn)(n), dtype=float)

    def sample_env(self, n: int, d: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n additive perturbation vectors ε_i ∈ R^d.

        Args:
            n (int): Number of samples.
            d (int): Dimensionality of the feature space.
            rng (np.random.Generator | None): Optional random number generator.

        Returns:
            np.ndarray: Array of shape (n, d).
        """
        if self._sample_env_fn is not None:
            return self._sample_env_fn(n, d)

        if rng is None:
            rng = np.random.default_rng()

        if self.env_mean is not None and len(self.env_mean) != d:
            msg = f"env_mean has length {len(self.env_mean)}, but d={d}. They must match."
            raise InvalidParameterError(msg)

        mean = self.env_mean if self.env_mean is not None else np.zeros(d)
        cov = self.env_cov if self.env_cov is not None else np.zeros((d, d))

        if np.allclose(cov, 0.0):
            return np.tile(mean, (n, 1))

        if cov.ndim == 1:
            return rng.normal(loc=mean, scale=np.sqrt(cov), size=(n, d))

        return rng.multivariate_normal(mean=mean, cov=cov, size=n)

    @property
    def params(self) -> dict[str, Any]:
        """Return the parameters of the distribution."""
        result = dict(self._params)
        if self.env_mean is not None:
            result["env_mean"] = self.env_mean
        if self.env_cov is not None:
            result["env_cov"] = self.env_cov
        return result
