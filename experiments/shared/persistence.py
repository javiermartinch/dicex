# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Artifact directory layout and serialization helpers for the experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

import yaml

# Hyperparameter-tuning run layout


@dataclass(frozen=True)
class RunLayout:
    """Filesystem layout for one hyperparameter-tuning execution mode."""

    root: Path
    models_root: Path
    gold_root: Path
    optuna_root: Path
    reports_root: Path
    plots_root: Path
    selected_configs_root: Path
    logs_root: Path


def experiment_layout(*, results_root: Path | None = None) -> RunLayout:
    """Return the stable artifact directory layout for execution.

    Args:
        results_root (Path | None): Optional custom results folder. If None, defaults to
            'experiments/hyperparameter_tuning/results'.

    Returns:
        RunLayout: A RunLayout object mapping the required directory paths.
    """
    if results_root is None:
        results_root = Path(__file__).resolve().parents[1] / "hyperparameter_tuning" / "results"
    root = results_root
    models_root = Path(__file__).resolve().parents[1] / "models"
    return RunLayout(
        root=root,
        models_root=models_root,
        gold_root=root / "gold_standards",
        optuna_root=root / "optuna",
        reports_root=root / "reports",
        plots_root=root / "reports" / "plots",
        selected_configs_root=root / "selected_configs",
        logs_root=root / "logs",
    )


def ensure_layout(layout: RunLayout, *, create_tuning_dirs: bool = False) -> None:
    """Create the directory layout structure if folders are missing.

    Args:
        layout (RunLayout): RunLayout mapping the folder structure to create.
        create_tuning_dirs (bool): Whether to force creation of tuning-specific directories.
            Defaults to False unless 'hyperparameter_tuning' is in layout.root path.
    """
    directories = [
        layout.root,
        layout.models_root,
        layout.logs_root,
    ]
    if create_tuning_dirs or "hyperparameter_tuning" in str(layout.root):
        directories.extend(
            [
                layout.reports_root,
                layout.plots_root,
                layout.gold_root,
                layout.optuna_root,
                layout.selected_configs_root,
            ]
        )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def gold_records_path(layout: RunLayout) -> Path:
    """Return the pickle artifact path for gold-standard records.

    Args:
        layout (RunLayout): RunLayout configuration.

    Returns:
        Path: The target Path of the gold-standard pickle file.
    """
    return layout.gold_root / "gold_standards.pkl"


def optuna_study_path(layout: RunLayout, bucket: str) -> Path:
    """Return the JSON file path for one Optuna study database.

    Args:
        layout (RunLayout): RunLayout configuration.
        bucket (str): Name of the dimension bucket.

    Returns:
        Path: The target Path for the study JSON file.
    """
    return layout.optuna_root / f"{bucket}_study.json"


def pareto_plot_path(layout: RunLayout, bucket: str) -> Path:
    """Return the HTML output path for one Pareto front visualization.

    Args:
        layout (RunLayout): RunLayout configuration.
        bucket (str): Name of the dimension bucket.

    Returns:
        Path: The target Path of the HTML plot.
    """
    return layout.plots_root / f"pareto_{bucket}.html"


def final_report_path(layout: RunLayout) -> Path:
    """Return the comparative markdown report path.

    Args:
        layout (RunLayout): RunLayout configuration.

    Returns:
        Path: The target Path of the Markdown report file.
    """
    return layout.reports_root / "pareto_front_report.md"


# Generic I/O helpers


def write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text to disk using a temporary file.

    Args:
        path (Path): Target Path to write to.
        content (str): String content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def write_json(path: Path, payload: Any) -> None:
    """Atomically serialize and write pretty JSON to disk.

    Args:
        path (Path): Target Path to write to.
        payload (Any): Python object to serialize.
    """
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def load_json(path: Path) -> Any:
    """Load and parse JSON file content from disk.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        Any: The deserialized JSON content.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_pickle(path: Path, payload: Any) -> None:
    """Serialize one Python object to a pickle file on disk.

    Args:
        path (Path): Target Path to write to.
        payload (Any): Python object to pickle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path) -> Any:
    """Load a pickled Python object from disk.

    Args:
        path (Path): Path of the pickle file.

    Returns:
        Any: The deserialized Python object.
    """
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write one dictionary payload as YAML to disk.

    Args:
        path (Path): Target Path to write the YAML file.
        payload (dict[str, Any]): Dictionary to dump.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


# Internal helpers


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    msg = f"Object of type {type(value).__name__} is not JSON serializable."
    raise TypeError(msg)
