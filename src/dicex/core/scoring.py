# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Unified model scoring with optional evaluation tracking.

Provides :class:`ModelScorer`, which dispatches predict / predict_proba
calls and optionally counts model evaluations per search phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import numpy as np

    from dicex.core.model_protocols import ClassifierModel, RegressorModel


class ModelScorer:
    """Score model predictions with optional evaluation tracking.

    When `track` is `True` the scorer counts how many individual
    predictions have been requested, both globally (`n_evals`) and
    per-phase (`phase_eval_counts`). Setting `track` to `False`
    (the default) disables all bookkeeping, giving a zero-overhead
    scoring wrapper suitable for standalone or public-API usage.

    Args:
        model (RegressorModel | ClassifierModel): The classifier or regressor model to score with.
        target_class (int | None): Optional class index to predict probabilities for (if classifier).
        track (bool): Whether to track the number of model evaluations.
    """

    def __init__(
        self,
        model: RegressorModel | ClassifierModel,
        target_class: int | None = None,
        *,
        track: bool = False,
    ) -> None:
        self.model = model
        self.target_class = target_class
        self._track = track

        # Tracking state (only meaningful when track=True)
        self.n_evals: int = 0
        self.phase_eval_counts: dict[str, int] = {}
        self._current_phase: str | None = None

        # Cached baseline score
        self._y0: float | None = None

    def score(self, points: np.ndarray) -> np.ndarray:
        """Score *points* and optionally count the evaluations.

        Args:
            points (np.ndarray): 2-D array of shape ``(n, d)``.

        Returns:
            np.ndarray: 1-D array of shape ``(n,)`` with model scores.
        """
        if self._track:
            n = len(points)
            self.n_evals += n
            if self._current_phase is not None:
                self.phase_eval_counts[self._current_phase] = self.phase_eval_counts.get(self._current_phase, 0) + n
        if self.target_class is not None:
            cls_model = cast("ClassifierModel", self.model)
            return cls_model.predict_proba(points)[:, self.target_class]
        reg_model = cast("RegressorModel", self.model)
        return reg_model.predict(points).flatten()

    def score_baseline(self, x0: np.ndarray) -> float:
        """Return and cache the model score at the baseline point ``x0``.

        The baseline evaluation is not counted towards ``n_evals`` because it
        is a free reference point, not a search evaluation.

        Args:
            x0 (np.ndarray): The baseline point.

        Returns:
            float: The model score at ``x0``.
        """
        if self._y0 is None:
            self._y0 = float(self.score(x0.reshape(1, -1))[0])
            if self._track:
                self.n_evals -= 1
                if self._current_phase is not None:
                    self.phase_eval_counts[self._current_phase] -= 1
        return self._y0

    def reset_baseline(self) -> None:
        """Invalidate the cached baseline so the next call recomputes it."""
        self._y0 = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Score predictions to satisfy the ``RegressorModel`` protocol.

        Args:
            x (np.ndarray): Input samples.

        Returns:
            np.ndarray: Model predictions.
        """
        return self.score(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities using the underlying classifier.

        Args:
            x (np.ndarray): Input samples.

        Returns:
            np.ndarray: Model probability predictions.
        """
        cls_model = cast("ClassifierModel", self.model)
        if self._track:
            n = len(x)
            self.n_evals += n
            if self._current_phase is not None:
                self.phase_eval_counts[self._current_phase] = self.phase_eval_counts.get(self._current_phase, 0) + n
        return cls_model.predict_proba(x)

    def get_y0(self) -> float:
        """Retrieve the cached baseline score.

        Returns:
            float: The cached baseline score.

        Raises:
            RuntimeError: If the baseline has not been set yet.
        """
        if self._y0 is None:
            msg = "Baseline not set. Call score_baseline(x0) first."
            raise RuntimeError(msg)
        return self._y0
