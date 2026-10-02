# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Perturbation-realization and CVaR heatmaps for a single DiCEx explanation.

The script runs DiCEx on one query point of a registered dataset with a pretrained
model, samples stochastic perturbation realizations around the optimal direction,
evaluates the pointwise improvement of each realization, and renders a two-panel
heatmap: the realized perturbation vectors next to their improvements, followed by
a summary row with the DiCEx direction and the empirical CVaR. The paper uses the
``cpu_act`` regression dataset with a random forest; ``run_all.sh`` passes the
exact arguments.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import TYPE_CHECKING, Any

import joblib
from matplotlib import patches
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
from sklearn.preprocessing import StandardScaler

# Ensure the repository and experiments roots are on sys.path (dicex comes from the installed environment)
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
for candidate in (REPO_ROOT, EXPERIMENTS_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dicex.core.cvar import lower_tail_cvar_empirical
from dicex.distributions.gaussian import GaussianPerturbation
from dicex.explainer import Dicex
from experiments.shared.datasets import build_raw_dataset, default_dataset_registry
from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.plots import setup_latex_matplotlib
from experiments.shared.query_points import select_query_points

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from experiments.shared.datasets import RawDataset

logger = get_logger("perturbation_heatmap")

# LaTeX display names for the features of the ``parkinsons`` dataset
PARKINSONS_FEATURE_MAP: dict[str, str] = {
    "x1": r"$\mathrm{MDVP:F_0}$",
    "x2": r"$\mathrm{MDVP:F_{hi}}$",
    "x3": r"$\mathrm{MDVP:F_{lo}}$",
    "x4": r"$\mathrm{Jitter(\%)}$",
    "x5": r"$\mathrm{Jitter(Abs)}$",
    "x6": r"$\mathrm{MDVP:RAP}$",
    "x7": r"$\mathrm{MDVP:PPQ}$",
    "x8": r"$\mathrm{Jitter:DDP}$",
    "x9": r"$\mathrm{Shimmer}$",
    "x10": r"$\mathrm{Shimmer(dB)}$",
    "x11": r"$\mathrm{APQ3}$",
    "x12": r"$\mathrm{APQ5}$",
    "x13": r"$\mathrm{MDVP:APQ}$",
    "x14": r"$\mathrm{DDA}$",
    "x15": r"$\mathrm{NHR}$",
    "x16": r"$\mathrm{HNR}$",
    "x17": r"$\mathrm{RPDE}$",
    "x18": r"$\mathrm{DFA}$",
    "x19": r"$\mathrm{spread1}$",
    "x20": r"$\mathrm{spread2}$",
    "x21": r"$\mathrm{D_2}$",
    "x22": r"$\mathrm{PPE}$",
}

# LaTeX display names for the features of the ``cpu_act`` dataset
CPU_ACT_NAMES: tuple[str, ...] = (
    "lread",
    "lwrite",
    "scall",
    "sread",
    "swrite",
    "fork",
    "exec",
    "rchar",
    "wchar",
    "pgout",
    "ppgout",
    "pgfree",
    "pgscan",
    "atch",
    "pgin",
    "ppgin",
    "pflt",
    "vflt",
    "runqsz",
    "freemem",
    "freeswap",
)

CPU_ACT_FEATURE_MAP: dict[str, str] = {
    "lread": r"$\mathrm{LogRead}$",
    "lwrite": r"$\mathrm{LogWrite}$",
    "scall": r"$\mathrm{SysCall}$",
    "sread": r"$\mathrm{SysRead}$",
    "swrite": r"$\mathrm{SysWrite}$",
    "fork": r"$\mathrm{Fork}$",
    "exec": r"$\mathrm{Exec}$",
    "rchar": r"$\mathrm{ReadChar}$",
    "wchar": r"$\mathrm{WriteChar}$",
    "pgout": r"$\mathrm{PageOut}$",
    "ppgout": r"$\mathrm{PagesOut}$",
    "pgfree": r"$\mathrm{PageFree}$",
    "pgscan": r"$\mathrm{PageScan}$",
    "atch": r"$\mathrm{Attach}$",
    "pgin": r"$\mathrm{PageIn}$",
    "ppgin": r"$\mathrm{PagesIn}$",
    "pflt": r"$\mathrm{ProtFault}$",
    "vflt": r"$\mathrm{VirtFault}$",
    "runqsz": r"$\mathrm{RunQueue}$",
    "freemem": r"$\mathrm{FreeMem}$",
    "freeswap": r"$\mathrm{FreeSwap}$",
}
for _i, _name in enumerate(CPU_ACT_NAMES, start=1):
    CPU_ACT_FEATURE_MAP[f"x{_i}"] = CPU_ACT_FEATURE_MAP[_name]
    CPU_ACT_FEATURE_MAP[f"feature_{_i - 1}"] = CPU_ACT_FEATURE_MAP[_name]


def load_dataset_and_model(dataset_name: str, model_family: str) -> tuple[RawDataset, Any, Any, tuple[str, ...]]:
    """Load the raw dataset and pretrained model artifact.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'cpu_act').
        model_family (str): Model family (e.g., 'random_forest', 'logistic', 'mlp', 'xgboost').

    Returns:
        tuple[RawDataset, Any, Any, tuple[str, ...]]:
            Raw dataset, trained model, scaler instance, and feature names.
    """
    specs = {s.name: s for s in default_dataset_registry()}
    if dataset_name not in specs:
        msg = f"Dataset '{dataset_name}' not found in registry. Available: {list(specs.keys())}"
        raise ValueError(msg)

    spec = specs[dataset_name]
    raw_ds = build_raw_dataset(spec)

    model_path = REPO_ROOT / "experiments" / "models" / f"{dataset_name}__{model_family}.joblib"
    if not model_path.exists():
        msg = f"Model artifact not found at {model_path}"
        raise FileNotFoundError(msg)

    data = joblib.load(model_path)
    model = data["model"]
    scaler = data.get("scaler")
    if scaler is None:
        scaler = StandardScaler().fit(raw_ds.x)

    if dataset_name == "cpu_act":
        feature_names = CPU_ACT_NAMES
    else:
        feature_names = tuple(data.get("feature_names", raw_ds.feature_names))

    return raw_ds, model, scaler, feature_names


def select_query_point(
    raw_ds: RawDataset,
    model: Any,
    scaler: Any,
    instance_idx: int | None = None,
    query_percentile: float | None = None,
    target_class: int = 1,
    task: str = "classification",
) -> tuple[int, NDArray[np.float64], NDArray[np.float64], float]:
    """Select the query point to explain.

    Args:
        raw_ds (RawDataset): Raw dataset.
        model (Any): Trained model.
        scaler (Any): Fitted feature scaler, or None.
        instance_idx (int | None): Explicit sample index; takes precedence when given.
        query_percentile (float | None): Percentile of the model output used to select the sample.
        target_class (int): Target class index for classification models.
        task (str): Task type ('regression' or 'classification').

    Returns:
        tuple[int, NDArray[np.float64], NDArray[np.float64], float]: Sample index, raw query
            point, scaled query point, and baseline model output.
    """
    x_raw = raw_ds.x
    x_scaled = scaler.transform(x_raw) if scaler is not None else x_raw

    if task == "regression":
        preds = model.predict(x_raw)
        if instance_idx is not None:
            idx = instance_idx
        elif query_percentile is not None:
            qps = select_query_points(x_raw, preds, percentiles=(query_percentile,))
            idx = qps[0].point_index
        else:
            # Default for regression: the sample with the highest prediction
            idx = int(np.argmax(preds))
        x0_raw = np.asarray(x_raw[idx], dtype=np.float64)
        x0_scaled = np.asarray(x_scaled[idx], dtype=np.float64)
        base_val = float(preds[idx])
        logger.info("Using instance %d: baseline predicted Y = %.4f", idx, base_val)
        return idx, x0_raw, x0_scaled, base_val

    probs = model.predict_proba(x_scaled)[:, target_class]

    if instance_idx is not None:
        idx = instance_idx
    elif query_percentile is not None:
        qps = select_query_points(x_raw, probs, percentiles=(query_percentile,))
        idx = qps[0].point_index
        logger.info(
            "Selected instance %d matching target percentile %.2f: predicted P(Y=%d|x0) = %.4f (target quantile = %.4f)",
            idx,
            query_percentile,
            target_class,
            probs[idx],
            float(np.quantile(probs, query_percentile)),
        )
    else:
        # Candidate points: class != target_class and low target probability
        candidates = np.where((raw_ds.y != target_class) & (probs < 0.25))[0]
        if len(candidates) == 0:
            candidates = np.where(raw_ds.y != target_class)[0]
        if len(candidates) == 0:
            candidates = np.argsort(probs)

        # Select candidate with lowest non-zero probability or minimum
        idx = int(candidates[np.argmin(probs[candidates])])

    x0_raw = np.asarray(x_raw[idx], dtype=np.float64)
    x0_scaled = np.asarray(x_scaled[idx], dtype=np.float64)
    base_prob = float(probs[idx])

    logger.info("Using instance %d: baseline P(Y=%d|x0) = %.4f", idx, target_class, base_prob)
    return idx, x0_raw, x0_scaled, base_prob


class ScaledClassifierWrapper:
    """Pass-through adapter that exposes only the ``predict_proba`` method of a classifier."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the class probabilities of the wrapped model.

        Args:
            x (NDArray[np.float64]): Input feature matrix.

        Returns:
            NDArray[np.float64]: Class probabilities.
        """
        return self.model.predict_proba(x)


class ScaledRegressorWrapper:
    """Adapter that maps scaled inputs back to raw space and returns the negated prediction.

    Negating the prediction turns load minimization into the maximization problem solved by DiCEx.
    """

    def __init__(self, model: Any, scaler: Any) -> None:
        self.model = model
        self.scaler = scaler

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the negated prediction of the wrapped model for scaled inputs.

        Args:
            x (NDArray[np.float64]): Scaled input features (1D or 2D).

        Returns:
            NDArray[np.float64]: Negated model predictions.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_raw = self.scaler.inverse_transform(x) if self.scaler is not None else x
        return -self.model.predict(x_raw)


def compute_dicex_explanation(
    model: Any,
    x0_scaled: NDArray[np.float64],
    perturbation: GaussianPerturbation,
    alpha: float = 0.10,
    preset: str = "low",
    seed: int = 42,
    task: str = "classification",
) -> tuple[NDArray[np.float64], float, float]:
    """Run DiCEx to obtain the optimal direction c*.

    Args:
        model (Any): Model operating on scaled inputs.
        x0_scaled (NDArray[np.float64]): Scaled query point.
        perturbation (GaussianPerturbation): Perturbation distribution.
        alpha (float): CVaR risk level.
        preset (str): DiCEx optimizer preset.
        seed (int): Random seed.
        task (str): Task type ('regression' or 'classification').

    Returns:
        tuple[NDArray[np.float64], float, float]: Optimal direction, its robust value, and the
            baseline (inaction) CVaR.
    """
    explainer = Dicex(
        model=model,
        task=task,
        target_class=1 if task == "classification" else None,
        perturbation=perturbation,
        alpha=alpha,
        scaler=None,
        preset=preset,
        verbose="none",
        seed=seed,
    )
    res = explainer.explain(x0_scaled)
    c_opt = res.direction
    robust_value = float(res.robust_value)
    baseline_cvar = float(res.metadata.get("baseline_cvar", 0.0))

    logger.info("DiCEx optimization complete: CVaR_%.2f = %.4f (baseline = %.4f)", alpha, robust_value, baseline_cvar)
    return c_opt, robust_value, baseline_cvar


def sample_perturbation_realizations(
    model: Any,
    x0_scaled: NDArray[np.float64],
    c_opt: NDArray[np.float64],
    perturbation: GaussianPerturbation,
    n_realizations: int,
    alpha: float,
    seed: int = 42,
    *,
    sort_by_functional: bool = True,
) -> dict[str, Any]:
    """Sample perturbation realizations and compute their pointwise improvements.

    Args:
        model (Any): Model operating on scaled inputs.
        x0_scaled (NDArray[np.float64]): Scaled query point.
        c_opt (NDArray[np.float64]): Optimal direction.
        perturbation (GaussianPerturbation): Perturbation distribution.
        n_realizations (int): Number of realizations to sample.
        alpha (float): CVaR risk level.
        seed (int): Random seed.
        sort_by_functional (bool): Whether to sort realizations by increasing improvement.

    Returns:
        dict[str, Any]: Realized perturbations, unit directions, improvements, empirical CVaR,
            tail size, baseline output, and realization order.
    """
    d = len(x0_scaled)
    rng = np.random.default_rng(seed)

    t_samples = perturbation.sample(n_realizations, d=d, rng=rng)
    env_samples = perturbation.sample_env(n_realizations, d=d, rng=rng)

    # Realized perturbation vectors: delta_i = T_i * c* + epsilon_i
    deltas = t_samples * c_opt.reshape(1, -1) + env_samples
    perturbed_points = x0_scaled.reshape(1, -1) + deltas

    if hasattr(model, "predict_proba"):
        y0 = float(model.predict_proba(x0_scaled.reshape(1, -1))[0, 1])
        y_pert = model.predict_proba(perturbed_points)[:, 1]
    else:
        y0 = float(model.predict(x0_scaled.reshape(1, -1))[0])
        y_pert = model.predict(perturbed_points)
    increments = np.asarray(y_pert - y0, dtype=np.float64)

    # Compute unit direction vectors: u_i = delta_i / ||delta_i||_2 in S^{d-1}
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized_deltas = deltas / norms

    # Order realizations
    if sort_by_functional:
        order = np.argsort(increments)
        deltas = deltas[order]
        normalized_deltas = normalized_deltas[order]
        increments = increments[order]
    else:
        order = np.arange(n_realizations)

    k_tail = max(1, math.ceil(alpha * n_realizations))
    sample_cvar = float(lower_tail_cvar_empirical(increments, alpha))

    return {
        "deltas": deltas,
        "normalized_deltas": normalized_deltas,
        "increments": increments,
        "sample_cvar": sample_cvar,
        "k_tail": k_tail,
        "y0": y0,
        "order": order.tolist(),
    }


def render_perturbation_heatmap(
    feature_names: tuple[str, ...],
    vars_matrix: NDArray[np.float64],
    increments: NDArray[np.float64],
    c_opt: NDArray[np.float64],
    sample_cvar: float,
    base_prob: float,
    alpha: float,
    k_tail: int,
    dataset_name: str,
    model_family: str,
    output_pdf: Path,
    output_png: Path,
    *,
    normalized: bool = True,
    titles_mode: str = "none",
    cmap_func: str = "YlGn",
    dpi: int = 300,
    vlim_delta_override: float | None = None,
    vmin_func_override: float | None = None,
    vmax_func_override: float | None = None,
) -> None:
    """Render and save the dual-panel publication heatmap.

    Args:
        feature_names (tuple[str, ...]): Names of features.
        vars_matrix (NDArray[np.float64]): Matrix of realized perturbation vectors (N x d),
            either unit directions (if normalized=True) or raw displacements (if normalized=False).
        increments (NDArray[np.float64]): Vector of realized functional increments (N,).
        c_opt (NDArray[np.float64]): Optimal nominal unit direction vector (d,).
        sample_cvar (float): Empirical CVaR of the sampled realizations.
        base_prob (float): Baseline model probability.
        alpha (float): Risk level alpha.
        k_tail (int): Number of realizations forming the alpha-tail.
        dataset_name (str): Dataset identifier.
        model_family (str): Model architecture name.
        output_pdf (Path): File path for PDF output.
        output_png (Path): File path for PNG output.
        normalized (bool): Whether vars_matrix represents unit direction vectors (default: True).
        titles_mode (str): Title display mode ('none', 'panels', 'all'). Default: 'none'.
        cmap_func (str): Matplotlib colormap name for the improvement column and the CVaR cell.
        dpi (int): Output raster DPI.
        vlim_delta_override (float | None): Optional override for symmetric feature delta scale limit.
        vmin_func_override (float | None): Optional override for minimum functional scale limit.
        vmax_func_override (float | None): Optional override for maximum functional scale limit.
    """
    plt, sns = setup_latex_matplotlib(font_scale=1.18)

    n_samples, d = vars_matrix.shape

    # Map feature names to formatted LaTeX if available
    if dataset_name == "cpu_act":
        feature_map = CPU_ACT_FEATURE_MAP
    elif dataset_name == "parkinsons":
        feature_map = PARKINSONS_FEATURE_MAP
    else:
        feature_map = {}
    display_names = [feature_map.get(name, name) for name in feature_names]

    # Symmetric scale for perturbation / direction vectors (centered at 0)
    abs_max_delta = max(
        float(np.abs(vars_matrix).max()),
        float(np.abs(c_opt).max()),
        0.5 if normalized else 0.2,
    )
    # Unit-direction components lie in [-1, 1]; a fixed scale keeps panels comparable across alphas.
    vlim_delta = 1.0 if normalized else math.ceil(abs_max_delta * 10) / 10.0
    if vlim_delta_override is not None:
        vlim_delta = vlim_delta_override

    # Functional scale: dynamically framed without clipping
    min_val = float(min(increments.min(), sample_cvar))
    max_val = float(max(increments.max(), sample_cvar))
    span = max_val - min_val
    padding = max(0.02, 0.08 * span)
    if dataset_name == "cpu_act":
        vmin_func = math.floor((min_val - padding) * 2) / 2.0
        vmax_func = math.ceil((max_val + padding) * 2) / 2.0
        if vmax_func - vmin_func < 0.5:
            vmax_func = vmin_func + 0.5
    else:
        vmin_func = max(-1.0, math.floor((min_val - padding) * 20) / 20.0)
        vmax_func = min(1.0, math.ceil((max_val + padding) * 20) / 20.0)
        if vmax_func - vmin_func < 0.08:
            vmax_func = min(1.0, vmin_func + 0.1)
            vmin_func = max(-1.0, vmax_func - 0.1)

    if vmin_func_override is not None:
        vmin_func = vmin_func_override
    if vmax_func_override is not None:
        vmax_func = vmax_func_override

    # Taller-than-wide proportion (portrait aspect ratio ~ 1.45 - 1.50)
    fig_width = 11.5
    if titles_mode == "none":
        top_margin_in = 0.35
        fig_height = max(12.5, 3.3 + 0.23 * n_samples)
    elif titles_mode == "panels":
        top_margin_in = 0.65
        fig_height = max(12.8, 3.6 + 0.23 * n_samples)
    else:  # "all"
        top_margin_in = 1.75
        fig_height = max(13.2, 4.3 + 0.23 * n_samples)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Font sizes
    ylabel_size = 22.0
    ytick_size = 14.5
    xlabel_size = 18.0
    xtick_size = 15.0
    summary_label_size = 17.5
    cvar_header_size = 17.0
    cbar_tick_size = 14.5
    cbar_label_size = 17.0
    legend_size = 18.5

    border_linewidth = 1.8

    bottom_margin_in = 2.40
    top_ratio = 1.0 - (top_margin_in / fig_height)
    bottom_ratio = bottom_margin_in / fig_height

    gs = GridSpec(
        nrows=2,
        ncols=3,
        height_ratios=[n_samples, 1.8],
        width_ratios=[d, 1.45, 0.40],  # [Variables, Functional, Colorbar functional]
        wspace=0.08,
        hspace=0.038,
        left=0.12,
        right=0.88,
        top=top_ratio,
        bottom=bottom_ratio,
    )

    ax_realizations_vars = fig.add_subplot(gs[0, 0])
    ax_realizations_func = fig.add_subplot(gs[0, 1], sharey=ax_realizations_vars)
    ax_cbar_func = fig.add_subplot(gs[0, 2])

    ax_summary_vars = fig.add_subplot(gs[1, 0])
    ax_summary_func = fig.add_subplot(gs[1, 1], sharey=ax_summary_vars)

    cmap_vars = "RdBu_r"

    # Top panel: realization heatmaps
    if normalized:
        title_vars = (
            r"\textbf{Realized Perturbation Directions} "
            r"$\frac{T_i \odot c^* + \varepsilon_i}{\|T_i \odot c^* + \varepsilon_i\|_2} \in \mathbb{S}^{d-1}$"
        )
        cbar_label = r"Normalized Direction Component $c_j$"
        summary_label = r"\textbf{DiCEx} $\mathbf{c}^*$"
    else:
        title_vars = r"\textbf{Realized Feature Perturbations} $T_i \odot c^* + \varepsilon_i$"
        cbar_label = r"Normalized Direction Component $c_j$"
        summary_label = r"\textbf{DiCEx} $\mathbf{c}^*$"

    if titles_mode in ("panels", "all"):
        ax_realizations_vars.set_title(
            title_vars,
            fontsize=17.5,
            pad=10,
        )
        ax_realizations_func.set_title(
            r"$M(x_0 + T_i \odot c^* + \varepsilon_i) - M(x_0)$",
            fontsize=17.5,
            pad=10,
        )

    sns.heatmap(
        vars_matrix,
        ax=ax_realizations_vars,
        cmap=cmap_vars,
        vmin=-vlim_delta,
        vmax=vlim_delta,
        center=0,
        cbar=False,
        linewidths=0.5,
        linecolor="#ffffff",
    )

    # Functional values heatmap (pure color mapping, without cell text annotations)
    sns.heatmap(
        increments.reshape(-1, 1),
        ax=ax_realizations_func,
        cmap=cmap_func,
        vmin=vmin_func,
        vmax=vmax_func,
        annot=False,
        cbar=True,
        cbar_ax=ax_cbar_func,
        linewidths=0.5,
        linecolor="#ffffff",
    )
    if dataset_name == "cpu_act":
        ax_cbar_func.set_ylabel(r"$\Delta \mathrm{CPU\ Load\ Reduction\ (\%)}$", fontsize=ylabel_size, labelpad=12)
    else:
        ax_cbar_func.set_ylabel(r"$M(x_0 + T \odot c^* + \varepsilon) - M(x_0)$", fontsize=ylabel_size, labelpad=12)
    ax_cbar_func.tick_params(labelsize=cbar_tick_size)

    # Highlight alpha-tail rows
    ax_realizations_vars.axhline(k_tail, color="#d9534f", linestyle="--", linewidth=2.2)
    ax_realizations_func.axhline(k_tail, color="#d9534f", linestyle="--", linewidth=2.2)

    # Y-axis labels for realizations: sparse integer indices (1, 5, 10, ...)
    tick_indices = [0] + list(range(4, n_samples, 5))
    if (n_samples - 1) not in tick_indices:
        tick_indices.append(n_samples - 1)
    ax_realizations_vars.set_yticks([i + 0.5 for i in tick_indices])
    ax_realizations_vars.set_yticklabels([str(i + 1) for i in tick_indices], rotation=0, fontsize=ytick_size)
    ax_realizations_vars.set_ylabel(
        r"Perturbation Realizations $(T_i, \varepsilon_i) \quad (i = 1 \dots n)$", fontsize=ylabel_size, labelpad=12
    )

    # Remove x-ticks on top panel
    ax_realizations_vars.set_xticks([])
    ax_realizations_vars.set_xlabel("")
    ax_realizations_func.set_xticks([])
    ax_realizations_func.set_xlabel("")
    plt.setp(ax_realizations_func.get_yticklabels(), visible=False)

    # Bottom panel: DiCEx summary row
    # The bottom feature cells display the optimal unit direction c*
    summary_directions = c_opt.reshape(1, -1)
    sns.heatmap(
        summary_directions,
        ax=ax_summary_vars,
        cmap=cmap_vars,
        vmin=-vlim_delta,
        vmax=vlim_delta,
        center=0,
        cbar=False,
        linewidths=0.5,
        linecolor="#ffffff",
    )

    # Highlight border around DiCEx variables row
    rect_vars = patches.Rectangle(
        (0, 0),
        d,
        1,
        linewidth=border_linewidth,
        edgecolor="#111111",
        facecolor="none",
        clip_on=False,
        zorder=10,
    )
    ax_summary_vars.add_patch(rect_vars)

    # Functional CVaR summary cell (pure color mapping, value in label below)
    sns.heatmap(
        np.array([[sample_cvar]]),
        ax=ax_summary_func,
        cmap=cmap_func,
        vmin=vmin_func,
        vmax=vmax_func,
        annot=False,
        cbar=False,
        linewidths=0.5,
        linecolor="#ffffff",
    )

    # Highlight border around CVaR cell
    rect_func = patches.Rectangle(
        (0, 0),
        1,
        1,
        linewidth=border_linewidth,
        edgecolor="#111111",
        facecolor="none",
        clip_on=False,
        zorder=10,
    )
    ax_summary_func.add_patch(rect_func)

    ax_summary_vars.set_yticks([0.5])
    ax_summary_vars.set_yticklabels([summary_label], rotation=0, fontsize=summary_label_size)
    ax_summary_vars.set_xticks(np.arange(d) + 0.5)
    ax_summary_vars.set_xticklabels(display_names, rotation=55, ha="right", fontsize=xtick_size)
    ax_summary_vars.set_xlabel(f"Dataset Variables ($d = {d}$)", fontsize=xlabel_size, labelpad=12)

    cvar_label_text = r"$\mathrm{CVaR}_\alpha$"
    ax_summary_func.set_xticks([0.5])
    ax_summary_func.set_xticklabels(
        [cvar_label_text], fontsize=cvar_header_size, fontweight="bold", rotation=55, ha="right"
    )
    plt.setp(ax_summary_func.get_yticklabels(), visible=False)

    # Horizontal colorbar dynamically centered under ax_summary_vars
    pos_vars = ax_summary_vars.get_position()
    center_vars_x = (pos_vars.x0 + pos_vars.x1) / 2.0
    cbar_width = 0.38
    cbar_left = center_vars_x - cbar_width / 2.0
    cbar_bottom = max(0.024, bottom_ratio * 0.16)
    cbar_height = max(0.013, 0.20 / fig_height)

    cbar_vars_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    norm_vars = Normalize(vmin=-vlim_delta, vmax=vlim_delta)
    cbar_vars = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_vars, cmap=cmap_vars),
        cax=cbar_vars_ax,
        orientation="horizontal",
    )
    cbar_vars.set_label(cbar_label, fontsize=cbar_label_size, labelpad=7)
    cbar_vars.ax.tick_params(labelsize=cbar_tick_size)

    # Alpha-tail legend in the bottom right corner with multiline value alongside alpha-tail threshold
    cvar_val_str = f"{sample_cvar:.2f}"
    legend_label = r"$\alpha\mbox{-tail threshold}$" + "\n" + rf"($\mathrm{{CVaR}}_\alpha = {cvar_val_str}$)"
    proxy_tail = Line2D(
        [0, 1],
        [0, 0],
        color="#d9534f",
        linestyle="--",
        linewidth=2.4,
        label=legend_label,
    )
    fig.legend(
        handles=[proxy_tail],
        loc="lower right",
        bbox_to_anchor=(0.94, cbar_bottom - 0.010),
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cccccc",
        framealpha=0.92,
        fontsize=legend_size,
        handlelength=2.2,
        borderpad=0.5,
    )

    if titles_mode == "all":
        model_title = model_family.replace("_", " ").title()
        ds_title = dataset_name.replace("_", " ").title()

        suptitle_y = 1.0 - (0.35 / fig_height)
        subtitle_y = 1.0 - (0.95 / fig_height)

        fig.suptitle(
            r"\textbf{DiCEx Stochastic Perturbation Realizations and CVaR Profile Heatmap}",
            fontsize=14.0,
            fontweight="bold",
            y=suptitle_y,
        )

        subtitle_text = (
            f"Dataset: \\textbf{{{ds_title}}} ($d={d}$) $\\quad\\vert\\quad$ "
            f"Model: \\textbf{{{model_title}}} $\\quad\\vert\\quad$ "
            f"Baseline $M(x_0) = {base_prob:.3f}$\n"
            f"$\\alpha = {alpha:.2f}$ (Tail: worst $k={k_tail}$ of $n={n_samples}$) $\\quad\\vert\\quad$ "
            f"$\\widehat{{\\Delta}}^{{\\mathrm{{CVaR}}}}_{{{alpha:.2f}}}(c^*) = {sample_cvar:+.3f}$"
        )
        fig.text(0.5, subtitle_y, subtitle_text, ha="center", va="center", fontsize=10.5, linespacing=1.35)

    # Save outputs
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", dpi=dpi)
    fig.savefig(output_png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    logger.info("Saved heatmap plots:\n  -> PDF: %s\n  -> PNG: %s", output_pdf, output_png)


def main() -> None:
    """Run the perturbation heatmap experiment (CLI entry point)."""
    parser = argparse.ArgumentParser(
        description="Generate DiCEx perturbation-realization and CVaR heatmaps for one query point."
    )
    parser.add_argument("--dataset", type=str, default="cpu_act", help="Dataset name from data/datasets.yaml.")
    parser.add_argument(
        "--model-family",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic", "mlp", "xgboost"],
        help="Model architecture family.",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="Lower-tail CVaR risk parameter alpha.")
    parser.add_argument("--n-realizations", type=int, default=40, help="Number of perturbation realizations N.")
    parser.add_argument("--mu", type=float, default=0.25, help="Mean directional perturbation mu.")
    parser.add_argument("--sigma", type=float, default=0.10, help="Directional perturbation standard deviation sigma.")
    parser.add_argument("--env-std", type=float, default=0.02, help="Environmental noise standard deviation.")
    parser.add_argument(
        "--instance-idx",
        type=int,
        default=None,
        help="Index of the query point (the paper uses 4928); selected automatically if omitted.",
    )
    parser.add_argument(
        "--query-percentile",
        type=float,
        default=None,
        help="Percentile of the model output used to select the query point when no index is given (e.g., 0.10).",
    )
    parser.add_argument("--preset", type=str, default="low", choices=["low", "mid", "high"], help="Optimizer preset.")
    parser.add_argument(
        "--mode",
        type=str,
        default="normalized",
        choices=["normalized", "raw", "both"],
        help="Plot mode: 'normalized' (unit directions, default), 'raw' (unnormalized displacements), or 'both'.",
    )
    parser.add_argument(
        "--titles",
        type=str,
        default="none",
        choices=["none", "panels", "all"],
        help="Title display mode: 'none' (default: no titles), 'panels' (panel headers only), or 'all' (full titles).",
    )
    parser.add_argument(
        "--cmap-func",
        type=str,
        default="YlGn",
        help="Colormap for the functional column (default: 'YlGn').",
    )
    parser.add_argument("--no-sort", action="store_true", help="Do not sort realizations by functional value.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "perturbation_heatmap" / "results"),
        help="Root output directory for results.",
    )

    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    plots_dir = results_dir / "plots"
    data_dir = results_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(script_name="perturbation_heatmap", log_dir=results_dir / "logs")

    logger.info(
        "Starting perturbation heatmap experiment on dataset=%s, model=%s, alpha=%.2f",
        args.dataset,
        args.model_family,
        args.alpha,
    )

    # Load the dataset and the pretrained model
    raw_ds, model, scaler, feature_names = load_dataset_and_model(args.dataset, args.model_family)
    d = len(feature_names)

    specs = {s.name: s for s in default_dataset_registry()}
    spec = specs[args.dataset]
    task = spec.task

    if task == "regression":
        wrapped_model = ScaledRegressorWrapper(model, scaler)
    else:
        wrapped_model = ScaledClassifierWrapper(model)

    # Select the query point
    idx, _x0_raw, x0_scaled, base_prob = select_query_point(
        raw_ds=raw_ds,
        model=model,
        scaler=scaler,
        instance_idx=args.instance_idx,
        query_percentile=args.query_percentile,
        target_class=1,
        task=task,
    )

    # Define the perturbation distribution
    perturbation = GaussianPerturbation(
        mu=args.mu,
        sigma=args.sigma,
        env_cov=np.full(d, args.env_std**2, dtype=float),
    )

    # Compute the DiCEx explanation
    c_opt, full_cvar, baseline_inaction = compute_dicex_explanation(
        model=wrapped_model,
        x0_scaled=x0_scaled,
        perturbation=perturbation,
        alpha=args.alpha,
        preset=args.preset,
        seed=args.seed,
        task=task,
    )

    # Sample realizations and evaluate their improvements
    sim_results = sample_perturbation_realizations(
        model=wrapped_model,
        x0_scaled=x0_scaled,
        c_opt=c_opt,
        perturbation=perturbation,
        n_realizations=args.n_realizations,
        alpha=args.alpha,
        seed=args.seed,
        sort_by_functional=not args.no_sort,
    )

    increments = sim_results["increments"]
    sample_cvar = sim_results["sample_cvar"]
    k_tail = sim_results["k_tail"]

    # Render and save the heatmaps
    if args.query_percentile is not None:
        p_tag = f"p{round(args.query_percentile * 100):02d}_"
    elif args.instance_idx is not None:
        p_tag = f"idx{args.instance_idx}_"
    else:
        p_tag = ""

    plot_stem = (
        f"heatmap_{args.dataset}_{args.model_family}_{p_tag}alpha{int(args.alpha * 100):02d}_n{args.n_realizations}"
    )

    # Normalized heatmap (unit direction vectors in S^{d-1})
    if args.mode in ("normalized", "both"):
        norm_pdf = plots_dir / f"{plot_stem}_normalized.pdf"
        norm_png = plots_dir / f"{plot_stem}_normalized.png"
        render_perturbation_heatmap(
            feature_names=feature_names,
            vars_matrix=sim_results["normalized_deltas"],
            increments=increments,
            c_opt=c_opt,
            sample_cvar=sample_cvar,
            base_prob=base_prob,
            alpha=args.alpha,
            k_tail=k_tail,
            dataset_name=args.dataset,
            model_family=args.model_family,
            output_pdf=norm_pdf,
            output_png=norm_png,
            normalized=True,
            titles_mode=args.titles,
            cmap_func=args.cmap_func,
        )
        default_pdf = plots_dir / f"{plot_stem}.pdf"
        default_png = plots_dir / f"{plot_stem}.png"
        shutil.copyfile(norm_pdf, default_pdf)
        shutil.copyfile(norm_png, default_png)

    # Raw heatmap (unnormalized feature displacements)
    if args.mode in ("raw", "both"):
        raw_pdf = plots_dir / f"{plot_stem}_raw.pdf"
        raw_png = plots_dir / f"{plot_stem}_raw.png"
        render_perturbation_heatmap(
            feature_names=feature_names,
            vars_matrix=sim_results["deltas"],
            increments=increments,
            c_opt=c_opt,
            sample_cvar=sample_cvar,
            base_prob=base_prob,
            alpha=args.alpha,
            k_tail=k_tail,
            dataset_name=args.dataset,
            model_family=args.model_family,
            output_pdf=raw_pdf,
            output_png=raw_png,
            normalized=False,
            titles_mode=args.titles,
            cmap_func=args.cmap_func,
        )
        if args.mode == "raw":
            default_pdf = plots_dir / f"{plot_stem}.pdf"
            default_png = plots_dir / f"{plot_stem}.png"
            shutil.copyfile(raw_pdf, default_pdf)
            shutil.copyfile(raw_png, default_png)

    # Save the JSON summary
    summary_record = {
        "dataset": args.dataset,
        "model_family": args.model_family,
        "instance_index": idx,
        "baseline_probability": base_prob,
        "alpha": args.alpha,
        "n_realizations": args.n_realizations,
        "k_tail": k_tail,
        "full_optimization_cvar": full_cvar,
        "baseline_inaction_cvar": baseline_inaction,
        "sample_realization_cvar": sample_cvar,
        "direction_vector_norm": float(np.linalg.norm(c_opt)),
        "increments_min": float(increments.min()),
        "increments_mean": float(increments.mean()),
        "increments_max": float(increments.max()),
        "increments_std": float(increments.std()),
        "feature_names": list(feature_names),
        "direction_c_opt": c_opt.tolist(),
        "tail_realization_indices": list(range(k_tail)),
        "increments": increments.tolist(),
    }
    json_path = data_dir / f"{plot_stem}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_record, f, indent=2)

    logger.info("Summary metadata saved -> %s", json_path)
    logger.info("Perturbation heatmap experiment finished.")


if __name__ == "__main__":
    main()
