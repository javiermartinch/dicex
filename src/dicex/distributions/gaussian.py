# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Gaussian perturbation distribution for the pair (T, ε).

The directional perturbation T is always R^d-valued. This single class covers:
- iid coordinates via scalar ``mu`` and scalar ``sigma``
- diagonal Gaussian laws via vector ``mu`` and vector ``sigma``
- full multivariate Gaussian laws via vector ``mu`` and matrix ``cov``
"""

from typing import Any, cast

import numpy as np

from dicex.distributions.base import BasePerturbation


class GaussianPerturbation(BasePerturbation):
    """Gaussian perturbation for the pair (T, ε).

    T is Gaussian directional execution noise.
    ε ~ N(env_mean, env_cov) is the additive environmental noise.
    """

    def __init__(  # noqa: PLR0913
        self,
        mu: float | np.ndarray,
        sigma: float | np.ndarray | None = None,
        *,
        cov: np.ndarray | None = None,
        d: int | None = None,
        env_mean: np.ndarray | None = None,
        env_cov: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize with Gaussian directional noise and additive noise.

        Args:
            mu (float | np.ndarray): Mean parameter of T. Scalar for shared mean,
                vector otherwise.
            sigma (float | np.ndarray | None): Standard deviation parameter of T.
                Scalar for iid noise, vector for diagonal covariance.
                Mutually exclusive with ``cov``.
            cov (np.ndarray | None): Covariance parameter of T. Can be diagonal
                ``(d,)`` or full covariance ``(d, d)``. Mutually exclusive with
                ``sigma``.
            d (int | None): Optional dimensionality of T when it cannot be
                inferred later.
            env_mean (np.ndarray | None): Mean vector m_ε of the additive noise
                (d,). If None, defaults to zero vector.
            env_cov (np.ndarray | None): Covariance matrix Σ_ε of the additive
                noise (d, d). If None, defaults to zero matrix.
            seed (int | None): Optional seed for the random number generator.
        """
        if (sigma is None) == (cov is None):
            msg = "Exactly one of 'sigma' or 'cov' must be provided."
            raise ValueError(msg)

        mu_arr = np.asarray(mu, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float) if sigma is not None else None
        cov_arr = np.asarray(cov, dtype=float) if cov is not None else None

        if sigma_arr is not None and np.any(sigma_arr < 0):
            msg = f"Standard deviation sigma cannot be negative. Got {sigma}"
            raise ValueError(msg)
        if cov_arr is not None:
            if cov_arr.ndim not in (1, 2):
                msg = f"cov must be a vector or matrix. Got shape {cov_arr.shape}"
                raise ValueError(msg)
            if cov_arr.ndim == 1 and np.any(cov_arr < 0):
                msg = "Diagonal covariance entries cannot be negative."
                raise ValueError(msg)

        self.mu = float(mu_arr) if mu_arr.ndim == 0 else mu_arr
        self.sigma = None if sigma_arr is None else (float(sigma_arr) if sigma_arr.ndim == 0 else sigma_arr)
        self.cov = cov_arr
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
        if isinstance(self.sigma, np.ndarray):
            return len(self.sigma)
        if self.cov is not None:
            return int(self.cov.shape[0])
        if self.env_mean is not None:
            return len(self.env_mean)
        if self.env_cov is not None:
            cov = self.env_cov
            return int(cov.shape[0] if cov.ndim == 2 else len(cov))  # noqa: PLR2004
        return 1

    def _mean_vector(self, dim: int) -> np.ndarray:
        if isinstance(self.mu, (int, float)):
            return np.full(dim, cast("float", self.mu))
        mu = np.asarray(self.mu, dtype=float)
        if len(mu) != dim:
            msg = f"mu has dimension {len(mu)} but expected {dim}."
            raise ValueError(msg)
        return mu

    def _sigma_vector(self, dim: int) -> np.ndarray:
        if self.sigma is None:
            msg = "sigma is not available when the Gaussian law is parameterized by cov."
            raise ValueError(msg)
        if isinstance(self.sigma, (int, float)):
            return np.full(dim, cast("float", self.sigma))
        sigma = np.asarray(self.sigma, dtype=float)
        if len(sigma) != dim:
            msg = f"sigma has dimension {len(sigma)} but expected {dim}."
            raise ValueError(msg)
        return sigma

    def sample(self, n: int, d: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n directional perturbation vectors from the configured Gaussian law.

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
        mu_vec = self._mean_vector(dim)
        if self.cov is not None:
            if self.cov.ndim == 1:
                if len(self.cov) != dim:
                    msg = f"cov has dimension {len(self.cov)} but expected {dim}."
                    raise ValueError(msg)
                return rng.normal(loc=mu_vec, scale=np.sqrt(self.cov), size=(n, dim))
            return rng.multivariate_normal(mean=mu_vec, cov=self.cov, size=n)

        sigma_vec = self._sigma_vector(dim)
        return rng.normal(loc=mu_vec, scale=sigma_vec, size=(n, dim))

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
        result: dict[str, Any] = {"mu": self.mu}
        if self.sigma is not None:
            result["sigma"] = self.sigma
        if self.cov is not None:
            result["cov"] = self.cov
        if self.d is not None:
            result["d"] = self.d
        if self.env_mean is not None:
            result["env_mean"] = self.env_mean
        if self.env_cov is not None:
            result["env_cov"] = self.env_cov
        return result
