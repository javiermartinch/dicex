# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Base class for perturbation distributions.

Every perturbation models the pair (T, ε) where:
    - T is a directional perturbation in R^d
    - ε is an additive multivariate perturbation (environmental noise)

The realized input is x₀ + T ⊙ c + ε.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from dicex.exceptions import InvalidParameterError


class BasePerturbation(ABC):
    """Abstract base class for all perturbation distributions.

    All perturbations model the joint pair (T, ε).
    T and ε are R^d-valued random vectors.

    Subclasses must set ``env_mean`` and ``env_cov`` attributes (or leave
    them as ``None`` for zero-noise defaults).
    """

    # Subclasses should set these; defaults give zero additive noise.
    env_mean: np.ndarray | None = None
    env_cov: np.ndarray | None = None

    @abstractmethod
    def sample(self, n: int, d: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n directional perturbation vectors T_i.

        Args:
            n (int): Number of samples.
            d (int | None): Optional feature-space dimension when the
                perturbation does not store it intrinsically.
            rng (np.random.Generator | None): Optional random number generator.

        Returns:
            np.ndarray: 2D array of shape (n, d).
        """

    def sample_env(self, n: int, d: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n additive perturbation vectors ε_i ∈ R^d.

        Args:
            n (int): Number of samples.
            d (int): Dimensionality of the feature space.
            rng (np.random.Generator | None): Optional random number generator.

        Returns:
            np.ndarray: 2D array of shape (n, d).
        """
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
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Return the parameters of the distribution."""

    def projected_env_mean(self, beta: np.ndarray) -> float:
        """Compute m_{β,ε} = β^T m_ε.

        Args:
            beta (np.ndarray): Model coefficients (1D array).

        Returns:
            float: The projected mean of the additive noise.
        """
        if self.env_mean is None:
            return 0.0
        return float(np.dot(beta, self.env_mean))

    def projected_env_var(self, beta: np.ndarray) -> float:
        """Compute v_{β,ε} = β^T Σ_ε β.

        Args:
            beta (np.ndarray): Model coefficients (1D array).

        Returns:
            float: The projected variance of the additive noise.
        """
        if self.env_cov is None:
            return 0.0
        cov = self.env_cov
        if cov.ndim == 1:
            return float(np.dot(beta**2, cov))
        return float(beta @ cov @ beta)
