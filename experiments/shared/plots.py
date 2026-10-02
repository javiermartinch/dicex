# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Matplotlib and seaborn configuration shared by the plotting scripts."""

from __future__ import annotations

import shutil
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def setup_matplotlib(font_scale: float = 1.05) -> tuple[Any, Any]:
    """Import and configure matplotlib/seaborn for non-interactive rendering.

    Args:
        font_scale (float): Font scaling factor for seaborn themes.

    Returns:
        tuple[Any, Any]: Tuple of (pyplot module, seaborn module).
    """
    mpl.use("Agg")

    sns.set_theme(style="whitegrid", font_scale=font_scale)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )
    return plt, sns


def setup_latex_matplotlib(font_scale: float = 1.05) -> tuple[Any, Any]:
    """Configure matplotlib with LaTeX rendering for publication-quality plots.

    Uses Computer Modern Roman font (matching LaTeX default), enables ``text.usetex``,
    and sets high DPI for saved figures. Requires a working LaTeX installation
    with ``pdflatex``, ``cm-super``, and ``amsmath`` packages.

    Args:
        font_scale (float): Font scaling factor for seaborn themes.

    Returns:
        tuple[Any, Any]: Tuple of (pyplot module, seaborn module).
    """
    mpl.use("Agg")

    sns.set_theme(style="whitegrid", font_scale=font_scale)

    # Fall back to a non-LaTeX serif font when pdflatex is unavailable
    _has_latex = shutil.which("pdflatex") is not None

    latex_rc: dict[str, Any] = {
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "font.family": "serif",
    }

    if _has_latex:
        latex_rc.update(
            {
                "text.usetex": True,
                "font.serif": ["Computer Modern Roman"],
            }
        )
    else:
        latex_rc["font.serif"] = ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"]

    plt.rcParams.update(latex_rc)
    return plt, sns
