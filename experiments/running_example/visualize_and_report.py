# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Visualization generator for the Credit Approval Score paper plots."""

import io
from pathlib import Path
import re
import sys
from typing import Any, Protocol

import cloudpickle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dicex.core.geometry import apply_directional_noise
from dicex.distributions.gaussian import GaussianPerturbation
from experiments.shared.helpers import repo_relative
from experiments.shared.log_config import configure_logging, get_logger
from experiments.shared.plots import setup_latex_matplotlib

FS_LABEL = 18
FS_LEGEND = 16

VMIN = -10.0
VMAX = 160.0


def _savefig_both(output_path: str, **kwargs: Any) -> None:
    """Save the current matplotlib figure as PDF.

    Args:
        output_path (str): File path where the PDF figure will be saved.
        **kwargs (Any): Additional keyword arguments passed to ``plt.savefig``.
    """
    kwargs.setdefault("transparent", True)
    plt.savefig(output_path, **kwargs)


class RegressionModel(Protocol):
    """Protocol for regression models that have a predict method."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict continuous target values.

        Args:
            x (np.ndarray): Input feature array.

        Returns:
            np.ndarray: Predicted target values.
        """
        ...


def plot_directional_comparison(
    x0: np.ndarray,
    x_nominal: np.ndarray | None,
    x_wachter: np.ndarray | None,
    directions_dicex: dict[float, np.ndarray],
    model: RegressionModel,
    output_path: str,
    *,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    x0_size: float = 300,
    arrow_width: float = 0.009,
    arrow_linewidth: float = 1.0,
    arrow_headwidth: float = 5.0,
    arrow_headlength: float = 6.0,
    legend_fontsize: float | None = None,
) -> None:
    """Plot the Wachter CF, nominal, and DiCEx(alpha) directions in a single figure.

    Args:
        x0 (np.ndarray): Query point in scaled space.
        x_nominal (np.ndarray | None): Optional nominal point.
        x_wachter (np.ndarray | None): Optional Wachter counterfactual point.
        directions_dicex (dict[float, np.ndarray]): Map from alpha to unit direction vector.
        model (RegressionModel): Regression model predicting continuous scores.
        output_path (str): File path for output figure.
        x_limits (tuple[float, float] | None): Range limits for x-axis. Defaults to None.
        y_limits (tuple[float, float] | None): Range limits for y-axis. Defaults to None.
        x0_size (float): Marker size for x0. Defaults to 300.
        arrow_width (float): Width of direction arrows. Defaults to 0.009.
        arrow_linewidth (float): Line width for arrow edges. Defaults to 1.0.
        arrow_headwidth (float): Head width for arrows. Defaults to 5.0.
        arrow_headlength (float): Head length for arrows. Defaults to 6.0.
        legend_fontsize (float | None): Legend font size. Defaults to None.
    """
    if x_limits is not None and y_limits is not None:
        x1_min, x1_max = x_limits
        x2_min, x2_max = y_limits
    else:
        margin = 1.0
        x1_min, x1_max = x0[0] - margin, x0[0] + margin
        x2_min, x2_max = x0[1] - margin, x0[1] + margin

    x1_grid = np.linspace(x1_min, x1_max, 300)
    x2_grid = np.linspace(x2_min, x2_max, 300)
    xx, yy = np.meshgrid(x1_grid, x2_grid)

    z_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z_pred = z_pred.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(
        xx,
        yy,
        z_pred,
        cmap="RdYlGn",
        alpha=0.8,
        shading="gouraud",
        rasterized=True,
        vmin=VMIN,
        vmax=VMAX,
    )
    cbar = plt.colorbar()
    cbar.set_label("Predicted Credit Score", fontsize=FS_LABEL)

    def _draw_arrow(c: np.ndarray, color: str, label: str) -> None:
        ax = plt.gca()
        ax.quiver(
            [x0[0]],
            [x0[1]],
            [c[0]],
            [c[1]],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            edgecolors="black",
            linewidths=arrow_linewidth,
            alpha=0.95,
            width=arrow_width,
            headwidth=arrow_headwidth,
            headlength=arrow_headlength,
            zorder=5,
            label=label,
        )

    dicex_colors = {0.1: "#e41a1c", 0.5: "#984ea3", 0.86: "#4daf4a", 0.9: "#984ea3", 1.0: "#4daf4a"}
    for alpha_val, c_dir in sorted(directions_dicex.items()):
        color = dicex_colors.get(alpha_val, "purple")
        _draw_arrow(c_dir, color, f"DiCEx $\\alpha={alpha_val:.2f}$")

    if x_nominal is not None:
        c_nominal = (x_nominal - x0) / np.linalg.norm(x_nominal - x0)
        _draw_arrow(c_nominal, "cyan", "Nominal (best deterministic)")

    if x_wachter is not None:
        c_cf = (x_wachter - x0) / np.linalg.norm(x_wachter - x0)
        _draw_arrow(c_cf, "lime", "Wachter CF")

    # Visible tags on direction arrows
    if x_wachter is not None:
        c_cf = (x_wachter - x0) / np.linalg.norm(x_wachter - x0)
        plt.text(
            x0[0] + c_cf[0] - 0.54,
            x0[1] + c_cf[1] - 0.50,
            r"Wachter CF",
            fontsize=14,
            color="#144814",
            ha="center",
            va="bottom",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lime", lw=1.6, alpha=0.95),
        )

    for alpha_val, c_dir in sorted(directions_dicex.items()):
        color = dicex_colors.get(alpha_val, "purple")
        if np.isclose(alpha_val, 0.9):
            plt.text(
                x0[0] + c_dir[0] + 0.08,
                x0[1] + c_dir[1] - 0.44,
                r"$\alpha = 0.90$",
                fontsize=14,
                color="#4a148c",
                ha="left",
                va="center",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.6, alpha=0.95),
            )
        elif np.isclose(alpha_val, 0.1):
            plt.text(
                -0.25,
                -1.00,
                r"$\alpha = 0.10$",
                fontsize=14,
                color="#b71c1c",
                ha="center",
                va="bottom",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.6, alpha=0.95),
            )
        else:
            plt.text(
                x0[0] + c_dir[0],
                x0[1] + c_dir[1],
                rf"$\alpha = {alpha_val:.2f}$",
                fontsize=14,
                color=color,
                ha="center",
                va="bottom",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.6, alpha=0.95),
            )

    plt.scatter([x0[0]], [x0[1]], c="black", s=x0_size, marker="*", edgecolors="black", zorder=6, label="$x_0$")

    plt.xlabel("Financial Capacity Index", fontsize=FS_LABEL)
    plt.ylabel("Home Value Index", fontsize=FS_LABEL)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    lg_fs = legend_fontsize if legend_fontsize is not None else FS_LEGEND
    plt.legend(fontsize=lg_fs, loc="lower right")
    plt.grid(visible=True, linestyle="--", alpha=0.3, color="0.35")
    plt.gca().set_axisbelow(False)

    _savefig_both(output_path, bbox_inches="tight")
    plt.close()


def plot_improvement_distribution_facets(
    imp_dicex: dict[float, np.ndarray],
    imp_wachter: np.ndarray | None,
    output_path: str,
    *,
    alphas_to_include: list[float] | None = None,
    include_wachter: bool = True,
    target_aspect: float | None = None,
) -> None:
    """Plot the improvement KDEs of the Wachter CF and DiCEx directions in a single panel.

    Args:
        imp_dicex (dict[float, np.ndarray]): Map from alpha to array of improvements.
        imp_wachter (np.ndarray | None): Array of Wachter CF improvements.
        output_path (str): File path for output figure.
        alphas_to_include (list[float] | None): List of alphas to plot. Defaults to None.
        include_wachter (bool): Whether to plot Wachter CF KDE. Defaults to True.
        target_aspect (float | None): Desired width/height aspect ratio. Defaults to None.
    """

    def _kde(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
        n = len(values)
        h = 1.06 * np.std(values) * n ** (-0.2)
        diff = x_grid[:, np.newaxis] - values[np.newaxis, :]
        kernel = np.exp(-0.5 * (diff / h) ** 2) / (h * np.sqrt(2 * np.pi))
        return kernel.mean(axis=1)

    all_arrays: list[np.ndarray] = []
    if include_wachter and imp_wachter is not None:
        all_arrays.append(imp_wachter)
    all_arrays.extend(
        [imp_dicex[a] for a in sorted(imp_dicex.keys()) if alphas_to_include is None or a in alphas_to_include]
    )
    all_data = np.concatenate(all_arrays)
    x_min, x_max = all_data.min(), all_data.max()
    x_pad = 0.1 * (x_max - x_min)
    x_plot = np.linspace(x_min - x_pad, x_max + x_pad, 500)

    fig, ax = plt.subplots(figsize=(10, 6))

    alpha_colors = {0.1: "#e41a1c", 0.5: "#984ea3", 0.9: "#984ea3", 1.0: "#377eb8"}
    alpha_linestyles = {0.1: "-", 0.5: "--", 0.9: "--", 1.0: ":"}

    for alpha_val in sorted(imp_dicex.keys()):
        if alphas_to_include is not None and alpha_val not in alphas_to_include:
            continue
        imp = imp_dicex[alpha_val]
        color = alpha_colors.get(alpha_val, "#333333")
        linestyle = alpha_linestyles.get(alpha_val, "-")

        kde_vals = _kde(imp, x_plot)
        ax.fill_between(x_plot, kde_vals, alpha=0.08, color=color)
        ax.plot(x_plot, kde_vals, color=color, linewidth=2.5, linestyle=linestyle, label=f"DiCEx $\\alpha={alpha_val}$")

    if include_wachter and imp_wachter is not None:
        kde_w = _kde(imp_wachter, x_plot)
        ax.fill_between(x_plot, kde_w, alpha=0.08, color="lime")
        ax.plot(x_plot, kde_w, color="lime", linewidth=2.5, linestyle=":", label="Wachter CF")

    ax.set_xlabel(r"Improvement $Z(c) = M(x_0 + T \odot c + \epsilon) - M(x_0)$", fontsize=FS_LABEL)
    ax.set_ylabel("Density", fontsize=FS_LABEL)

    ax.legend(fontsize=FS_LEGEND, loc="upper left", framealpha=0.85)

    sm = ScalarMappable(cmap="RdYlGn")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.15, pad=0.04)
    cbar.ax.set_visible(False)

    ax.grid(visible=True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    if target_aspect is not None:
        for _ in range(5):
            buf = io.BytesIO()
            fig.savefig(buf, format="pdf", bbox_inches="tight", transparent=True)
            match = re.search(
                rb"/MediaBox\s*\[\s*([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)\s*\]", buf.getvalue()
            )
            if match:
                x1, y1, x2, y2 = [float(v) for v in match.groups()]
                w_cur, h_cur = x2 - x1, y2 - y1
                cur_aspect = w_cur / h_cur
                if abs(cur_aspect - target_aspect) < 1e-4:
                    break
                w_fig, h_fig = fig.get_size_inches()
                fig.set_size_inches(w_fig * (target_aspect / cur_aspect), h_fig)

    _savefig_both(output_path, bbox_inches="tight")
    plt.close()


def plot_cf_direction(
    x0: np.ndarray,
    cf_point: np.ndarray,
    c_cf: np.ndarray,
    pts_cf: np.ndarray | None,
    model: RegressionModel,
    output_path: str,
    *,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    highlight_failures: bool = False,
    cf_as_arrow: bool = False,
    plot_cf_point: bool = False,
    arrow_color: str = "navy",
    arrow_label: str | None = None,
    cf_label: str = r"$x_{\mathrm{CF}}$",
    samples_label: str | None = None,
    legend_fontsize: float | None = None,
    x0_size: float = 300,
    cf_size: float = 150,
    samples_size: float = 32,
    arrow_width: float = 0.009,
) -> str:
    r"""Plot the Wachter CF direction together with its noisy realizations.

    Args:
        x0 (np.ndarray): Query point in scaled space.
        cf_point (np.ndarray): The Wachter CF point (deterministic).
        c_cf (np.ndarray): Unit direction vector for Wachter CF.
        pts_cf (np.ndarray | None): Noisy sample points along the CF direction.
        model (RegressionModel): Regression model predicting continuous scores.
        output_path (str): File path for output figure.
        x_limits (tuple[float, float] | None): Range limits for x-axis. Defaults to None.
        y_limits (tuple[float, float] | None): Range limits for y-axis. Defaults to None.
        highlight_failures (bool): Whether to color failures in red. Defaults to False.
        cf_as_arrow (bool): Draw counterfactual vector as arrow. Defaults to False.
        plot_cf_point (bool): Plot counterfactual point. Defaults to False.
        arrow_color (str): Arrow color. Defaults to "navy".
        arrow_label (str | None): Legend label for arrow. Defaults to None.
        cf_label (str): Legend label for CF point. Defaults to "$x_{\mathrm{CF}}$".
        samples_label (str | None): Legend label for sample points. Defaults to None.
        legend_fontsize (float | None): Legend font size. Defaults to None.
        x0_size (float): Marker size for x0. Defaults to 300.
        cf_size (float): Marker size for CF point. Defaults to 150.
        samples_size (float): Marker size for noise samples. Defaults to 32.
        arrow_width (float): Width of arrow. Defaults to 0.009.

    Returns:
        str: The output file path.
    """
    if x_limits is not None and y_limits is not None:
        x1_min, x1_max = x_limits
        x2_min, x2_max = y_limits
    else:
        margin = 1.5
        x1_min, x1_max = x0[0] - margin, x0[0] + margin
        x2_min, x2_max = x0[1] - margin, x0[1] + margin

    x1_grid = np.linspace(x1_min, x1_max, 300)
    x2_grid = np.linspace(x2_min, x2_max, 300)
    xx, yy = np.meshgrid(x1_grid, x2_grid)
    z_pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 6))

    plt.pcolormesh(
        xx,
        yy,
        z_pred,
        cmap="RdYlGn",
        alpha=0.8,
        shading="gouraud",
        rasterized=True,
        vmin=VMIN,
        vmax=VMAX,
    )
    cbar = plt.colorbar()
    cbar.set_label("Predicted Credit Score", fontsize=FS_LABEL)

    if pts_cf is not None:
        if highlight_failures:
            y0 = model.predict(x0.reshape(1, -1))[0]
            y_pts = model.predict(pts_cf)

            failures_mask = y_pts < y0
            successes_mask = ~failures_mask

            if np.any(successes_mask):
                plt.scatter(
                    pts_cf[successes_mask, 0],
                    pts_cf[successes_mask, 1],
                    c="#3498DB",
                    s=samples_size,
                    alpha=1.0,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=4,
                    label="Success",
                )
            if np.any(failures_mask):
                plt.scatter(
                    pts_cf[failures_mask, 0],
                    pts_cf[failures_mask, 1],
                    c="#E74C3C",
                    s=samples_size * 1.3,
                    alpha=1.0,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                    label="Failure",
                )
        else:
            scatter_kwargs = {}
            if samples_label:
                scatter_kwargs["label"] = samples_label
            plt.scatter(
                pts_cf[:, 0],
                pts_cf[:, 1],
                c="#3498DB",
                s=samples_size,
                alpha=1.0,
                edgecolors="black",
                linewidths=0.3,
                zorder=4,
                **scatter_kwargs,
            )

    if cf_as_arrow:
        ax = plt.gca()
        dx = c_cf[0]
        dy = c_cf[1]
        quiver_kwargs = {}
        if arrow_label:
            quiver_kwargs["label"] = arrow_label
        ax.quiver(
            [x0[0]],
            [x0[1]],
            [dx],
            [dy],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=arrow_color,
            edgecolors="black",
            linewidths=1.0,
            alpha=0.95,
            width=arrow_width,
            headwidth=5.0,
            headlength=6.0,
            zorder=5,
            **quiver_kwargs,
        )

    if not cf_as_arrow or plot_cf_point:
        plt.scatter(
            cf_point[0],
            cf_point[1],
            c="lime",
            s=cf_size,
            marker="D",
            edgecolors="black",
            zorder=6,
            label=cf_label,
        )

    plt.scatter(
        [x0[0]],
        [x0[1]],
        c="black",
        s=x0_size,
        marker="*",
        edgecolors="black",
        zorder=6,
        label=r"$x_0$",
    )

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xlabel("Financial Capacity Index", fontsize=FS_LABEL)
    plt.ylabel("Home Value Index", fontsize=FS_LABEL)
    l_size = legend_fontsize if legend_fontsize is not None else FS_LEGEND
    plt.legend(loc="lower right", fontsize=l_size)
    plt.grid(visible=True, linestyle="--", alpha=0.3, color="0.35")
    plt.gca().set_axisbelow(False)

    _savefig_both(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def create_vector_field_plot(
    x_grid_1: np.ndarray,
    x_grid_2: np.ndarray,
    directions_u: np.ndarray,
    directions_v: np.ndarray,
    mask: np.ndarray,
    no_move_mask: np.ndarray | None,
    model: Any,
    output_path: str,
    *,
    x_limits: tuple[float, float] = (-2.0, 2.0),
    y_limits: tuple[float, float] = (-2.5, 2.5),
    show_colorbar: bool = True,
    show_ylabel: bool = True,
) -> None:
    """Create a quiver plot with the model surface as background.

    Args:
        x_grid_1 (np.ndarray): Meshgrid X1 coordinates.
        x_grid_2 (np.ndarray): Meshgrid X2 coordinates.
        directions_u (np.ndarray): U components of vector field.
        directions_v (np.ndarray): V components of vector field.
        mask (np.ndarray): Boolean mask indicating active vector points.
        no_move_mask (np.ndarray | None): Boolean mask for inaction points.
        model (Any): Regression model.
        output_path (str): File path for output figure.
        x_limits (tuple[float, float]): Range limits for x-axis. Defaults to (-2.0, 2.0).
        y_limits (tuple[float, float]): Range limits for y-axis. Defaults to (-2.5, 2.5).
        show_colorbar (bool): Whether to include colorbar. Defaults to True.
        show_ylabel (bool): Whether to show y-axis label. Defaults to True.
    """
    fig_w, fig_h = 7.8, 7.0
    ax_left, ax_bottom, ax_w, ax_h = 1.2, 0.9, 4.5, 5.4

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([ax_left / fig_w, ax_bottom / fig_h, ax_w / fig_w, ax_h / fig_h])

    x1_bg = np.linspace(x_limits[0], x_limits[1], 400)
    x2_bg = np.linspace(y_limits[0], y_limits[1], 400)
    xx_bg, yy_bg = np.meshgrid(x1_bg, x2_bg)
    z_bg = model.predict(np.c_[xx_bg.ravel(), yy_bg.ravel()]).reshape(xx_bg.shape)

    norm_bg = Normalize(vmin=z_bg.min(), vmax=z_bg.max())
    mesh = ax.pcolormesh(xx_bg, yy_bg, z_bg, cmap="RdYlGn", alpha=0.8, shading="gouraud", norm=norm_bg, rasterized=True)
    if show_colorbar:
        cax = fig.add_axes([(ax_left + ax_w + 0.15) / fig_w, ax_bottom / fig_h, 0.25 / fig_w, ax_h / fig_h])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("Credit Approval Score", fontsize=18)
        cbar.ax.tick_params(labelsize=14)

    x1_flat = x_grid_1[mask]
    x2_flat = x_grid_2[mask]
    u_flat = directions_u[mask]
    v_flat = directions_v[mask]

    ax.quiver(
        x1_flat,
        x2_flat,
        u_flat,
        v_flat,
        angles="xy",
        scale_units="xy",
        scale=3.0,
        color="navy",
        alpha=0.85,
        width=0.005,
        headwidth=4,
        headlength=5,
        zorder=5,
    )

    if no_move_mask is not None and no_move_mask.any():
        x1_no = x_grid_1[no_move_mask]
        x2_no = x_grid_2[no_move_mask]
        ax.scatter(
            x1_no,
            x2_no,
            marker="x",
            c="red",
            s=80,
            linewidths=1.5,
            zorder=6,
            label="Inaction",
        )
        ax.legend(fontsize=22, loc="lower right")

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel("Financial Capacity Index", fontsize=18)
    if show_ylabel:
        ax.set_ylabel("Home Value Index", fontsize=18)
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=14)
    ax.grid(visible=True, linestyle="--", alpha=0.3, color="0.35")
    ax.set_axisbelow(False)

    _savefig_both(output_path, dpi=600, transparent=True)
    plt.close(fig)


def plot_vector_field_maps(
    vector_field_data: dict[str, Any],
    model_scaled: Any,
    output_dir: Path,
) -> None:
    """Generate vector field map plots from precomputed data.

    Args:
        vector_field_data (dict[str, Any]): Precomputed vector field dictionary.
        model_scaled (Any): Scaled wrapper model.
        output_dir (Path): Output directory path for saving plots.
    """
    xx1 = vector_field_data["xx1"]
    xx2 = vector_field_data["xx2"]
    dicex_directions = vector_field_data["dicex_directions"]
    u_cf = vector_field_data["u_cf"]
    v_cf = vector_field_data["v_cf"]
    ok_cf = vector_field_data["ok_cf"]
    x_limits = vector_field_data.get("x_limits", (-2.0, 2.0))
    y_limits = vector_field_data.get("y_limits", (-2.5, 2.5))

    for alpha, (u, v, ok, no_move) in dicex_directions.items():
        alpha_str = str(alpha).replace(".", "_")
        show_cb = abs(alpha - 0.1) < 1e-3
        create_vector_field_plot(
            xx1,
            xx2,
            u,
            v,
            ok,
            no_move,
            model_scaled,
            str(output_dir / f"vector_field_dicex_alpha_{alpha_str}.pdf"),
            x_limits=x_limits,
            y_limits=y_limits,
            show_colorbar=show_cb,
            show_ylabel=False,
        )

    create_vector_field_plot(
        xx1,
        xx2,
        u_cf,
        v_cf,
        ok_cf,
        None,
        model_scaled,
        str(output_dir / "vector_field_wachter_cf.pdf"),
        x_limits=x_limits,
        y_limits=y_limits,
        show_colorbar=False,
        show_ylabel=True,
    )


def main() -> None:
    """Generate the running-example figures from the serialized pipeline state (CLI entry point)."""
    setup_latex_matplotlib(font_scale=1.3)

    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    plots_dir = base_dir / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(script_name="visualize_and_report", log_dir=base_dir / "results" / "logs")
    logger = get_logger("visualize_and_report")

    data_path = results_dir / "running_example_data.pkl"
    if not data_path.exists():
        logger.error("Data file not found: %s. Run running_example.py first.", repo_relative(data_path))
        sys.exit(1)

    logger.info("Loading pipeline data from %s", repo_relative(data_path))
    with open(data_path, "rb") as f:
        data = cloudpickle.load(f)

    x0_scaled = data["x0_scaled"]
    cf_scaled = data["cf_scaled"]
    model_scaled = data["model_scaled"]
    directions = data["directions"]
    pts_wachter = data.get("pts_wachter")
    y0_val = data["y0_val"]

    if cf_scaled is not None:
        logger.info("Generating x0_wachter_cf_scaled.pdf...")
        plot_cf_direction(
            x0=x0_scaled,
            cf_point=cf_scaled,
            c_cf=data["c_cf"],
            pts_cf=None,
            model=model_scaled,
            output_path=str(plots_dir / "x0_wachter_cf_scaled.pdf"),
            x_limits=(-1.5, 2.0),
            y_limits=(-2.0, 2.5),
            cf_as_arrow=False,
            plot_cf_point=True,
            legend_fontsize=18,
        )

    if cf_scaled is not None and pts_wachter is not None:
        logger.info("Generating x0_wachter_cf_direction_samples_scaled.pdf...")
        plot_cf_direction(
            x0=x0_scaled,
            cf_point=cf_scaled,
            c_cf=data["c_cf"],
            pts_cf=pts_wachter,
            model=model_scaled,
            output_path=str(plots_dir / "x0_wachter_cf_direction_samples_scaled.pdf"),
            x_limits=(-1.5, 2.0),
            y_limits=(-2.0, 2.5),
            cf_as_arrow=True,
            plot_cf_point=False,
            arrow_color="lime",
            arrow_label=r"$c$",
            samples_label="Realizations",
            legend_fontsize=18,
        )

    if cf_scaled is not None and 0.1 in directions and 0.9 in directions:
        logger.info("Generating directional_comparison.pdf...")
        directions_to_compare = {0.1: directions[0.1], 0.9: directions[0.9]}
        plot_directional_comparison(
            x0=x0_scaled,
            x_nominal=None,
            x_wachter=cf_scaled if cf_scaled is not None else None,
            directions_dicex=directions_to_compare,
            model=model_scaled,
            output_path=str(plots_dir / "directional_comparison.pdf"),
            x_limits=(-1.5, 2.0),
            y_limits=(-2.0, 2.5),
            x0_size=300,
            arrow_width=0.009,
            arrow_linewidth=1.0,
            arrow_headwidth=5.0,
            arrow_headlength=6.0,
            legend_fontsize=18,
        )

    if cf_scaled is not None and 0.1 in directions and 0.9 in directions:
        logger.info("Generating improvement_distribution.pdf...")
        dynamic_mu = float(np.linalg.norm(cf_scaled - x0_scaled))
        pert_step = GaussianPerturbation(
            mu=dynamic_mu * 1.7,
            sigma=0.3,
            env_mean=np.array([0.0, 0.0]),
            env_cov=np.array([0.1, 0.1]),
            seed=42,
        )
        n_samples_kde = 10000
        noise_v_kde = pert_step.sample(n_samples_kde)
        env_v_kde = pert_step.sample_env(n_samples_kde, d=2)

        c_cf_vec = data["c_cf"]
        pts_var_wachter_kde = apply_directional_noise(x0_scaled, c_cf_vec, noise_v_kde, env_v_kde)
        pts_var_dicex_0_9_kde = apply_directional_noise(x0_scaled, directions[0.9], noise_v_kde, env_v_kde)
        pts_var_dicex_0_1_kde = apply_directional_noise(x0_scaled, directions[0.1], noise_v_kde, env_v_kde)

        imp_dicex_0_1 = model_scaled.predict(pts_var_dicex_0_1_kde) - y0_val
        imp_dicex_0_9 = model_scaled.predict(pts_var_dicex_0_9_kde) - y0_val
        imp_wachter_step = model_scaled.predict(pts_var_wachter_kde) - y0_val

        imp_dicex_seq = {0.1: imp_dicex_0_1, 0.9: imp_dicex_0_9}

        plot_improvement_distribution_facets(
            imp_dicex=imp_dicex_seq,
            imp_wachter=imp_wachter_step,
            output_path=str(plots_dir / "improvement_distribution.pdf"),
            alphas_to_include=[0.1, 0.9],
            include_wachter=True,
            target_aspect=614.04 / 394.09,
        )

    if "vector_fields" in data:
        logger.info("Generating vector field maps...")
        plot_vector_field_maps(data["vector_fields"], model_scaled, plots_dir)

    logger.info("Running-example figures written to %s", repo_relative(plots_dir))


if __name__ == "__main__":
    main()
