# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Test that the public API is correctly exported."""

import numpy as np

import dicex


def test_public_api_exports() -> None:
    """Verify that all main components are accessible from the root package."""
    assert hasattr(dicex, "Dicex")
    assert hasattr(dicex, "GaussianPerturbation")
    assert hasattr(dicex, "UniformPerturbation")
    assert hasattr(dicex, "ExplanationResult")
    assert hasattr(dicex, "__version__")


def test_version_is_valid() -> None:
    """Verify that __version__ is a string and not empty."""
    assert isinstance(dicex.__version__, str)
    assert len(dicex.__version__) > 0


def test_explanation_result_default_metadata() -> None:
    """Verify ExplanationResult uses default_factory for metadata."""
    direction = np.array([1.0, 0.0])
    result = dicex.ExplanationResult(direction=direction, robust_value=0.5, alpha=0.1)
    assert isinstance(result.metadata, dict)
    assert len(result.metadata) == 0


def test_explanation_result_explicit_metadata() -> None:
    """Verify ExplanationResult accepts explicit metadata."""
    direction = np.array([1.0, 0.0])
    meta = {"iterations": 42}
    result = dicex.ExplanationResult(direction=direction, robust_value=0.5, alpha=0.1, metadata=meta)
    assert result.metadata == meta
