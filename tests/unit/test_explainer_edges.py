# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Edge-case tests for the explainer API."""

import numpy as np
import pytest

from dicex.exceptions import InvalidModelError, InvalidParameterError, NotFittedError
from dicex.explainer import Dicex


class DummyModel:
    pass


class DummyRegressor:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)


def test_explainer_invalid_task() -> None:
    """Verify exception on invalid task."""
    with pytest.raises(InvalidParameterError, match="Unsupported task"):
        Dicex(model=DummyModel(), task="magic")  # type: ignore[arg-type]


def test_explainer_classification_missing_target() -> None:
    """Verify target_class is required for classification."""
    with pytest.raises(InvalidParameterError, match="target_class must be provided"):
        Dicex(model=DummyModel(), task="classification")


def test_explainer_invalid_model_methods() -> None:
    """Verify exception when model lacks required methods."""
    with pytest.raises(InvalidModelError, match="'predict_proba' method"):
        Dicex(model=DummyModel(), task="classification", target_class=1)

    with pytest.raises(InvalidModelError, match="'predict' method"):
        Dicex(model=DummyModel(), task="regression")


def test_explainer_fit_invalid_shape() -> None:
    """Verify fit checks 2D shape."""
    explainer = Dicex(model=DummyRegressor(), task="regression", scaler="auto")
    with pytest.raises(InvalidParameterError, match="must be a 2D array"):
        explainer.fit(np.array([1.0, 2.0]))  # 1D instead of 2D


def test_explainer_explain_not_fitted() -> None:
    """Verify explain requires fit when scaler='auto'."""
    explainer = Dicex(model=DummyRegressor(), task="regression", scaler="auto")
    with pytest.raises(NotFittedError, match=r"Dicex\.fit must be called"):
        explainer.explain(np.array([1.0, 2.0]))


def test_explainer_explain_invalid_shape() -> None:
    """Verify explain checks 1D shape."""
    explainer = Dicex(model=DummyRegressor(), task="regression", scaler=None)
    with pytest.raises(InvalidParameterError, match="must be a 1D array"):
        explainer.explain(np.array([[1.0, 2.0]]))  # 2D instead of 1D


def test_explainer_explain_batch_invalid_shape() -> None:
    """Verify explain_batch checks 2D shape."""
    explainer = Dicex(model=DummyRegressor(), task="regression", scaler=None)
    with pytest.raises(InvalidParameterError, match="must be a 2D array"):
        explainer.explain_batch(np.array([1.0, 2.0]))  # 1D instead of 2D


def test_explainer_rejects_unsupported_scaler() -> None:
    """A non-DicexScaler scaler (e.g., sklearn's StandardScaler) raises a clear error."""
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    with pytest.raises(InvalidParameterError, match="Unsupported scaler of type 'StandardScaler'"):
        Dicex(model=DummyRegressor(), scaler=StandardScaler())  # type: ignore[arg-type]
    with pytest.raises(InvalidParameterError, match="Unsupported scaler of type 'str'"):
        Dicex(model=DummyRegressor(), scaler="standard")  # type: ignore[arg-type]


def test_explainer_accepts_converted_sklearn_scaler() -> None:
    """The conversion suggested in the error message yields a usable, already fitted scaler."""
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    from dicex.preprocessing.scaling import DicexScaler  # noqa: PLC0415

    x = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
    sk = StandardScaler()
    sk.fit(x)  # pyright: ignore[reportUnknownMemberType]
    scaler = DicexScaler()
    scaler.mean_ = np.asarray(sk.mean_, dtype=float)  # pyright: ignore[reportUnknownMemberType]
    scaler.std_ = np.asarray(sk.scale_, dtype=float)
    explainer = Dicex(model=DummyRegressor(), scaler=scaler)
    assert explainer.scaler is scaler
    expected = np.asarray(sk.transform(x), dtype=float)  # pyright: ignore[reportUnknownMemberType]
    assert np.allclose(scaler.transform(x), expected)
