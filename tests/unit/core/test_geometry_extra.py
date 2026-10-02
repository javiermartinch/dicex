# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Extreme edge cases for geometry functions."""

import numpy as np
import pytest

from dicex.core.geometry import apply_directional_noise, orthogonal_unit_vector


def test_apply_directional_noise_1d_conversion() -> None:
    """Verify 1D noise array is converted to 2D."""
    x0 = np.array([1.0])
    direction = np.array([1.0])
    noise = np.array([2.0, 3.0])  # 1D array
    points = apply_directional_noise(x0, direction, noise)
    assert points.shape == (2, 1)
    assert np.allclose(points, [[3.0], [4.0]])


def test_apply_directional_noise_invalid_ndim() -> None:
    """Verify ValueError for >2D noise array."""
    x0 = np.array([1.0])
    direction = np.array([1.0])
    noise = np.zeros((2, 1, 1))  # 3D array
    with pytest.raises(ValueError, match="noise_samples must have shape"):
        apply_directional_noise(x0, direction, noise)


def test_apply_directional_noise_mismatched_dim() -> None:
    """Verify ValueError when noise dimension doesn't match x0."""
    x0 = np.array([1.0, 2.0])
    direction = np.array([1.0, 0.0])
    noise = np.zeros((2, 3))  # 3 features instead of 2 or 1
    with pytest.raises(ValueError, match="second dimension must match"):
        apply_directional_noise(x0, direction, noise)


def test_orthogonal_unit_vector_failure() -> None:
    """Verify ValueError when orthogonal vector construction fails."""
    # In one dimension no nonzero vector is orthogonal to v = [1.0]: the canonical basis
    # vector e = [1.0] is parallel to v, so its orthogonal residual vanishes.
    with (
        np.errstate(divide="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="Cannot construct a stable orthogonal direction"),
    ):
        orthogonal_unit_vector(np.array([1.0]))
