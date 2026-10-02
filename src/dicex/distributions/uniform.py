# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Uniform perturbation distribution for the pair (T, ε).

The directional perturbation T is R^d-valued with iid uniform coordinates.
"""

from typing import Any, cast

import numpy as np

from dicex.distributions.base import BasePerturbation


class UniformPerturbation(BasePerturbation):
    """Uniform perturbation for the pair (T, ε).

    T has iid coordinates distributed as Unif[mu - delta, mu + delta].
    ε ~ N(env_mean, env_cov) is the additive environmental noise.
    """

    def __init__(
        self,
        mu: float | np.ndarray,
        delta: float | np.ndarray,
        *,
        d: int | None = None,
        env_mean: np.ndarray | None = None,
        env_cov: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize with scalar noise parameters and additive noise parameters.

        Args:
            mu (float | np.ndarray): Mean of the distribution.
            delta (float | np.ndarray): Half-width of the distribution (radius).
            d (int | None): Optional dimensionality of T when it cannot be
                inferred later.
            env_mean (np.ndarray | None): Mean vector m_ε of the additive noise
                (d,). If None, defaults to zero vector.
            env_cov (np.ndarray | None): Covariance matrix Σ_ε of the additive
                noise (d, d). If None, defaults to zero matrix.
            seed (int | None): Optional seed for the random number generator.
        """
        mu_scalar = isinstance(mu, (int, float))
        delta_scalar = isinstance(delta, (int, float))
        delta_arr = np.asarray(delta, dtype=float)
        if np.any(delta_arr < 0):
            msg = f"Half-width delta cannot be negative. Got {delta}"
            raise ValueError(msg)
        scalar_mu = float(cast("float", mu)) if mu_scalar else None
        scalar_delta = float(cast("float", delta)) if delta_scalar else None
        self.mu = scalar_mu if scalar_mu is not None else np.asarray(mu, dtype=float)
        self.delta = scalar_delta if scalar_delta is not None else delta_arr
        self.d = d
        self.env_mean = np.asarray(env_mean, dtype=float) if env_mean is not None else None
        self.env_cov = np.asarray(env_cov, dtype=float) if env_cov is not None else None
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _resolve_dim(self, d: int | None = None) -> int:  # noqa: PLR0911
        if self.d is not None:
            return self.d
        if d is not None:
            return d
        if isinstance(self.mu, np.ndarray):
            return len(self.mu)
        if isinstance(self.delta, np.ndarray):
            return len(self.delta)
        if self.env_mean is not None:
            return len(self.env_mean)
        if self.env_cov is not None:
            cov = self.env_cov
            return int(cov.shape[0] if cov.ndim == 2 else len(cov))  # noqa: PLR2004
        return 1

    def sample(self, n: int, d: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n directional perturbation vectors with iid uniform coordinates.

        Args:
            n (int): Number of samples.
            d (int | None): Optional feature-space dimension.
            rng (np.random.Generator | None): Optional random number generator.

        Returns:
            np.ndarray: 2D array of shape (n, dim).
        """
        if rng is None:
            rng = self._rng
        dim = self._resolve_dim(d)
        mu_vec = np.full(dim, cast("float", self.mu)) if isinstance(self.mu, (int, float)) else self.mu
        delta_vec = np.full(dim, cast("float", self.delta)) if isinstance(self.delta, (int, float)) else self.delta
        return rng.uniform(low=mu_vec - delta_vec, high=mu_vec + delta_vec, size=(n, dim))

    def sample_env(self, n: int, d: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n additive perturbation vectors ε_i from N(m_ε, Σ_ε).

        Args:
            n (int): Number of samples.
            d (int): Dimensionality of the feature space.
            rng (np.random.Generator | None): Optional random number generator.

        Returns:
            np.ndarray: 2D array of shape (n, d).
        """
        if rng is None:
            rng = self._rng
        return super().sample_env(n, d, rng)

    @property
    def params(self) -> dict[str, Any]:
        """Return the parameters of the distribution.

        Returns:
            dict[str, Any]: Dictionary containing the perturbation configuration.
        """
        result: dict[str, Any] = {"mu": self.mu, "delta": self.delta}
        if self.d is not None:
            result["d"] = self.d
        if self.env_mean is not None:
            result["env_mean"] = self.env_mean
        if self.env_cov is not None:
            result["env_cov"] = self.env_cov
        return result
