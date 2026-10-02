# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""High-level explainer API for DiCEx.

This module exposes the task-oriented ``Dicex`` facade, with explicit fit and
explain steps and optional automatic feature scaling.
"""

from pathlib import Path
import time
from typing import Any, Literal, Self, cast

import numpy as np

from dicex.adapters.sklearn import wrap_sklearn_classifier, wrap_sklearn_regressor
from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.core.presets import PRESETS
from dicex.distributions.base import BasePerturbation
from dicex.distributions.gaussian import GaussianPerturbation
from dicex.exceptions import InvalidModelError, InvalidParameterError, NotFittedError
from dicex.logging import configure_verbosity, get_logger
from dicex.optim.vmf_vns import VmfVns
from dicex.preprocessing.scaling import DicexScaler
from dicex.results import ExplanationResult

VECTOR_NDIM = 1
MATRIX_NDIM = 2
DIM_LOW = 10
DIM_MEDIUM = 50
TaskType = Literal["regression", "classification"]

_logger = get_logger("explainer")


class Dicex:
    """Directional counterfactual explainer for a fitted model.

    Configure the explainer with the model, the uncertainty of the execution and the
    risk level, call `fit` on training data to learn the standardization of the
    features, and call `explain` (or `explain_batch`) to obtain the recommended
    direction of change for a record.
    """

    def __init__(  # noqa: PLR0913
        self,
        model: Any,  # noqa: ANN401
        task: TaskType = "regression",
        target_class: int | None = None,
        perturbation: BasePerturbation | None = None,
        alpha: float = 0.1,
        scaler: DicexScaler | Literal["auto"] | None = "auto",
        preset: Literal["low", "mid", "high"] = "mid",
        *,
        verbose: bool | str | int = "info",
        log_file: str | Path | None = None,
        **vmf_vns_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the high-level DiCEx explainer.

        Args:
            model (Any): Fitted model to explain, exposing ``predict`` (regression) or
                ``predict_proba`` (classification).
            task (TaskType): ``"regression"`` raises the prediction; ``"classification"`` raises the
                probability of ``target_class``.
            target_class (int | None): Index of the class whose probability is raised. Required for
                classification.
            perturbation (BasePerturbation | None): Law of the random step length ``T`` along the
                direction and of the additive noise. Defaults to
                ``GaussianPerturbation(mu=0.1, sigma=0.02)``.
                Its parameters are expressed in standardized units (standard deviations of each
                feature) when a scaler is used, and in the original feature units when ``scaler=None``.
            alpha (float): Risk level of the lower-tail CVaR objective: ``1.0`` maximizes the
                expected gain, ``0.1`` the mean of the worst 10% of outcomes.
            scaler (DicexScaler | Literal["auto"] | None):
                ``"auto"`` learns a scaler from ``fit``, or pass a fitted ``DicexScaler``. Use ``None`` to
                search and perturb directly in the original feature units. In all cases, the returned
                direction is expressed in the original feature space.
            preset (Literal["low", "mid", "high"]): Computational budget of the optimizer, adapted
                to the number of features (see ``dicex.core.presets.PRESETS``).
            verbose (bool | str | int): ``"info"`` (default), ``"debug"``, ``"none"``, or a
                ``logging`` level.
            log_file (str | Path | None): If provided, also write the log to this file.
            **vmf_vns_kwargs (Any): Options of `VmfVns` that override the preset, e.g. ``seed``
                for reproducible results, ``max_evals`` or ``timeout_seconds``.
        """
        if task not in ("regression", "classification"):
            msg = f"Unsupported task '{task}'. Expected 'regression' or 'classification'."
            raise InvalidParameterError(msg)
        if task == "classification" and target_class is None:
            msg = "target_class must be provided for classification tasks."
            raise InvalidParameterError(msg)

        self.original_model = model
        self.task = task
        self.target_class = target_class
        self.alpha = alpha
        self.perturbation = perturbation or GaussianPerturbation(mu=0.1, sigma=0.02)
        self.preset = preset
        self._vmf_vns_kwargs = dict(vmf_vns_kwargs)
        self.verbose = verbose
        self.log_file = log_file
        self._d: int | None = None

        # Configure logging
        if verbose or log_file:
            configure_verbosity(verbose=verbose, log_file=log_file)

        self.model = self._adapt_model(model)

        if scaler == "auto":
            self.scaler: DicexScaler | None = DicexScaler()
            self._is_fitted = False
        elif scaler is None:
            self.scaler = None
            self._is_fitted = True
        elif isinstance(scaler, DicexScaler):  # pyright: ignore[reportUnnecessaryIsInstance]
            self.scaler = scaler
            self._is_fitted = scaler.mean_ is not None and scaler.std_ is not None
        else:
            msg = (
                f"Unsupported scaler of type {type(scaler).__name__!r}. Use scaler='auto' (a DicexScaler is "
                "fitted in `fit`), scaler=None (no scaling), or a dicex.preprocessing.scaling.DicexScaler instance. "
                "To reuse a fitted sklearn StandardScaler `s`, create `d = DicexScaler()` and set "
                "`d.mean_ = s.mean_` and `d.std_ = s.scale_`."
            )
            raise InvalidParameterError(msg)

    def _adapt_model(self, model: Any) -> Any:  # noqa: ANN401
        """Adapt the incoming model to the internal prediction protocols.

        Args:
            model (Any): The machine learning model to adapt.

        Returns:
            Any: The adapted model wrapping the original.

        Raises:
            InvalidModelError: If the model lacks the required prediction methods.
        """
        if self.task == "classification":
            if not hasattr(model, "predict_proba"):
                msg = "Classification models must provide a 'predict_proba' method."
                raise InvalidModelError(msg)
            return wrap_sklearn_classifier(model)

        if not hasattr(model, "predict"):
            msg = "Regression models must provide a 'predict' method."
            raise InvalidModelError(msg)
        return wrap_sklearn_regressor(model)

    def fit(self, x_train: np.ndarray) -> Self:
        """Fit the internal scaler when scaling is enabled.

        Args:
            x_train (np.ndarray): Training data array of shape (n_samples, n_features).

        Returns:
            Self: The fitted explainer instance.

        Raises:
            InvalidParameterError: If `x_train` is not a 2D array.
        """
        if x_train.ndim != MATRIX_NDIM:
            msg = f"x_train must be a 2D array. Got shape {x_train.shape}"
            raise InvalidParameterError(msg)

        if self.scaler is not None:
            self.scaler.fit(x_train)
            _logger.info("Scaler fitted on %d samples, d=%d", len(x_train), x_train.shape[1])

        self._d = x_train.shape[1]
        self._is_fitted = True
        return self

    def _get_dimension_bucket(self) -> str:
        """Categorize the problem dimension into a bucket for preset lookup.

        Returns:
            str: The dimension bucket name ("low", "medium", or "high").
        """
        if self._d is None:
            return "medium"  # pragma: no cover
        if self._d <= DIM_LOW:
            return "low"
        if self._d <= DIM_MEDIUM:
            return "medium"
        return "high"

    def _optimizer_params(self) -> dict[str, Any]:
        """Build the optimizer parameter dictionary with project defaults.

        Returns:
            dict[str, Any]: A dictionary of parameters ready to be passed to `VmfVns`.
        """
        bucket = self._get_dimension_bucket()
        preset_params = PRESETS[bucket][self.preset]

        defaults: dict[str, Any] = {
            "kappas": [1.0, 10.0, 100.0, 1000.0],
            "b_samples": preset_params["b_samples"],
            "n_exploration": preset_params["n_exploration"],
            "n_test": preset_params["n_test"],
            "n_evaluation": preset_params["n_evaluation"],
            "n_bootstrap": preset_params["n_bootstrap"],
            "eta": preset_params["eta"],
            "grid_t": np.linspace(0, 1, preset_params["grid_t_len"]),
            "max_evals": None,
            "timeout_seconds": None,
            "min_runtime_seconds": None,
            "n_restarts": 0,
            "trace_level": "none",
            "seed": None,
        }
        defaults.update(self._vmf_vns_kwargs)
        return defaults

    def explain(self, x0: np.ndarray) -> ExplanationResult:  # noqa: PLR0915
        """Generate a directional counterfactual explanation for a single point.

        Args:
            x0 (np.ndarray): The baseline point to explain, as a 1D array.

        Returns:
            ExplanationResult: The optimal direction and metrics.

        Raises:
            NotFittedError: If the explainer requires fitting but `fit` has not been called.
            InvalidParameterError: If `x0` is not a 1D array.
        """
        if not self._is_fitted:
            msg = "Dicex.fit must be called before explain when scaler='auto'."
            raise NotFittedError(msg)

        if x0.ndim != VECTOR_NDIM:
            msg = f"x0 must be a 1D array. Got shape {x0.shape}"
            raise InvalidParameterError(msg)

        if self._d is None:
            self._d = len(x0)

        _logger.info(
            "Explaining point: d=%d, alpha=%.3f, task=%s",
            len(x0),
            self.alpha,
            self.task,
        )

        model_for_optimizer = self.model
        x0_optimizer = x0.astype(float, copy=False)
        metadata: dict[str, Any] = {}
        perturbation_for_optimizer = self.perturbation

        if self.scaler is not None:
            x0_optimizer = self.scaler.transform(x0.reshape(1, -1)).reshape(-1)
            model_for_optimizer = self.scaler.wrap_model(self.model)

        optimizer_params = self._optimizer_params()

        optimizer = VmfVns(
            model=cast("Any", model_for_optimizer),
            x0=x0_optimizer,
            perturbation=cast("BasePerturbation", perturbation_for_optimizer),
            alpha=self.alpha,
            target_class=self.target_class,
            **cast("Any", optimizer_params),
        )
        result_scaled = optimizer.run()

        direction = result_scaled.direction
        if self.scaler is not None:
            direction = self.scaler.transform_direction(result_scaled.direction)
            metadata["direction_scaled"] = result_scaled.direction

        metadata.update(result_scaled.metadata)

        _logger.info(
            "Explanation complete: CVaR=%.6f, %d model evals",
            result_scaled.robust_value,
            result_scaled.metadata.get("n_model_evals", 0),
        )

        # Compute baseline CVaR (c=0) with the same reevaluation budget.
        n_baseline = int(optimizer_params["n_evaluation"])
        baseline_t0 = time.perf_counter()
        env_samples = cast("BasePerturbation", perturbation_for_optimizer).sample_env(n_baseline, len(x0_optimizer))
        pts_baseline = x0_optimizer + env_samples

        if self.target_class is not None:
            cls_model = cast("Any", model_for_optimizer)
            y_baseline = cls_model.predict_proba(pts_baseline)[:, self.target_class]
            y0 = cls_model.predict_proba(x0_optimizer.reshape(1, -1))[0, self.target_class]
        else:
            reg_model = cast("Any", model_for_optimizer)
            y_baseline = reg_model.predict(pts_baseline).flatten()
            y0 = reg_model.predict(x0_optimizer.reshape(1, -1)).flatten()[0]

        z_baseline = y_baseline - float(y0)
        baseline_cvar = float(lower_tail_cvar_empirical(z_baseline, self.alpha))
        metadata["baseline_cvar"] = baseline_cvar
        phase_eval_counts = dict(metadata.get("phase_eval_counts", {}))
        phase_wall_times = dict(metadata.get("phase_wall_times", {}))
        phase_eval_counts["baseline_evaluation"] = phase_eval_counts.get("baseline_evaluation", 0) + n_baseline + 1
        phase_wall_times["baseline_evaluation"] = round(
            phase_wall_times.get("baseline_evaluation", 0.0) + (time.perf_counter() - baseline_t0),
            6,
        )
        metadata["phase_eval_counts"] = phase_eval_counts
        metadata["phase_wall_times"] = phase_wall_times
        metadata.setdefault("n_test", optimizer_params["n_test"])
        metadata.setdefault("n_evaluation", n_baseline)
        metadata.setdefault("n_bootstrap", optimizer_params.get("n_bootstrap", 1000))
        metadata.setdefault("trace_level", optimizer_params.get("trace_level", "none"))

        if result_scaled.robust_value > baseline_cvar:
            _logger.info(
                "Action recommended: Found direction strictly better than inaction (%.4f > %.4f)",
                result_scaled.robust_value,
                baseline_cvar,
            )
        else:
            _logger.info(
                "Inaction recommended: Best direction is not better than absorbing ambient noise (%.4f <= %.4f)",
                result_scaled.robust_value,
                baseline_cvar,
            )

        return ExplanationResult(
            direction=direction,
            robust_value=result_scaled.robust_value,
            alpha=self.alpha,
            metadata=metadata,
        )

    def explain_batch(self, x: np.ndarray) -> list[ExplanationResult]:
        """Explain each row of a 2D input array independently.

        Args:
            x (np.ndarray): Array of points to explain, of shape (n_samples, n_features).

        Returns:
            list[ExplanationResult]: A list of explanation results.

        Raises:
            InvalidParameterError: If `x` is not a 2D array.
        """
        if x.ndim != MATRIX_NDIM:
            msg = f"x must be a 2D array. Got shape {x.shape}"
            raise InvalidParameterError(msg)

        _logger.info("Batch explanation: %d points", len(x))
        return [self.explain(row) for row in x]
