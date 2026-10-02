# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Dataset registry, OpenML loading, and local caching for the benchmark datasets."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
# The package ``dicex`` is imported from the installed environment; the repository root is
# only added so that the ``experiments`` package can be imported.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataclasses import dataclass
import json
import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import openml
import pandas as pd
import yaml

if TYPE_CHECKING:
    from experiments.shared.specs import DimensionBucket, TaskType

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
OPENML_DATA_HOME = Path(__file__).resolve().parents[1] / "data" / "raw"
INTERIM_DATA_HOME = Path(__file__).resolve().parents[1] / "data" / "interim"
DATASETS_YAML_PATH = Path(__file__).resolve().parents[1] / "data" / "datasets.yaml"


class DatasetLoadError(RuntimeError):
    """Raised when a configured real dataset cannot be materialized."""


@dataclass(frozen=True)
class RawDataset:
    """Materialized dataset before model fitting and query-point selection."""

    x: FloatArray
    y: FloatArray | IntArray
    feature_names: tuple[str, ...]
    source: str
    notes: str


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata for one benchmark dataset."""

    name: str
    task: TaskType
    dimension_bucket: DimensionBucket
    n_features: int
    source: str
    description: str = ""
    url: str | None = None

    # Optional model and builder identifiers
    model_family: str | None = None
    builder_key: str | None = None

    # OpenML loader fields
    loader_key: str | None = None
    positive_class_label: int | str | None = None
    openml_candidates: tuple[str, ...] = ()
    openml_id: int | None = None
    target_column: str | None = None
    is_multiclass: bool = False


def _validate_numeric_feature_matrix(x_data: FloatArray, *, dataset_name: str) -> None:
    """Assert that the input feature matrix consists solely of continuous, numeric data.

    Checks that there are no non-numeric columns and no constant or binary columns,
    which are invalid for the perturbation-based explanation method.

    Args:
        x_data (FloatArray): Feature matrix as a 2D numpy array.
        dataset_name (str): Name of the dataset for error reporting.

    Raises:
        ValueError: If non-numeric, binary, or constant columns are detected.
    """
    if not np.issubdtype(x_data.dtype, np.number):
        msg = f"Dataset '{dataset_name}' must contain only numeric feature columns."
        raise ValueError(msg)

    invalid_columns = [index for index in range(x_data.shape[1]) if np.unique(x_data[:, index]).size <= 2]
    if invalid_columns:
        msg = f"Dataset '{dataset_name}' contains binary or constant feature columns: {invalid_columns}."
        raise ValueError(msg)


def _load_openml(spec: DatasetSpec) -> RawDataset:
    """Load a dataset from OpenML, downloading and caching locally if not found.

    Args:
        spec (DatasetSpec): Static metadata for the dataset.

    Returns:
        RawDataset: A RawDataset object containing features, targets, and names.

    Raises:
        DatasetLoadError: If retrieval or cache parsing fails.
    """
    raw_dir = OPENML_DATA_HOME / spec.dimension_bucket
    parquet_path = raw_dir / f"{spec.name}.parquet"
    meta_path = raw_dir / f"{spec.name}_metadata.json"

    interim_dir = INTERIM_DATA_HOME / spec.dimension_bucket
    x_path = interim_dir / f"{spec.name}_X.npy"
    y_path = interim_dir / f"{spec.name}_y.npy"

    if not (parquet_path.exists() and meta_path.exists()):
        if spec.openml_id is None:
            msg = f"Dataset '{spec.name}' is missing raw parquet files and has no 'openml_id' configured to download."
            raise DatasetLoadError(msg)

        logger.info("Downloading raw dataset '%s' (OpenML ID: %s) from OpenML...", spec.name, spec.openml_id)
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            dataset = openml.datasets.get_dataset(
                spec.openml_id, download_data=True, download_qualities=False, download_features_meta_data=False
            )
            df, _, _, _ = dataset.get_data(dataset_format="dataframe")

            # Convert sparse columns to dense (pyarrow does not support pandas SparseDtype)
            sparse_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.SparseDtype)]
            for c in sparse_cols:
                df[c] = df[c].sparse.to_dense()

            # Save raw dataset to parquet
            df.to_parquet(parquet_path, index=False)

            metadata = {
                "name": spec.name,
                "openml_id": spec.openml_id,
                "target_column": spec.target_column or dataset.default_target_attribute,
                "n_samples": len(df),
                "n_features_raw": len(df.columns) - 1,
            }
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as exc:
            msg = f"Failed to download or cache OpenML dataset '{spec.name}' (ID: {spec.openml_id})."
            raise DatasetLoadError(msg) from exc

    if x_path.exists() and y_path.exists():
        x_data = np.load(x_path, allow_pickle=True)
        y_data = np.load(y_path, allow_pickle=True)

        if spec.task == "classification" and not np.issubdtype(y_data.dtype, np.integer):
            y_data = _binarize_target(y_data, positive_class_label=spec.positive_class_label)
            np.save(y_path, y_data)

        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        target_col = spec.target_column or metadata.get("target_column")
        df_cols = [c for c in pd.read_parquet(parquet_path).columns if target_col is None or c != target_col]
        feature_names = tuple(df_cols)
        if len(feature_names) != x_data.shape[1]:
            feature_names = tuple(f"feature_{i}" for i in range(x_data.shape[1]))

        return RawDataset(
            x=x_data,
            y=y_data,
            feature_names=feature_names,
            source=f"interim_cache:{spec.name}",
            notes=f"Loaded from interim numpy cache for '{spec.name}'.",
        )

    try:
        df = pd.read_parquet(parquet_path)
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        msg = f"Failed to read dataset '{spec.name}' from raw cache."
        raise DatasetLoadError(msg) from exc

    target_col = spec.target_column or metadata.get("target_column")
    if not target_col or target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in dataset '{spec.name}'."
        raise DatasetLoadError(msg)

    try:
        y_raw = df[target_col].to_numpy()
        df_features = df.drop(columns=[target_col])

        # Drop non-numeric / categorical columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

        # Drop constant columns
        valid_cols = []
        for col in numeric_cols:
            col_data = df_features[col].to_numpy()
            if np.unique(col_data[~np.isnan(col_data)]).size > 1:
                valid_cols.append(col)

        df_clean = df_features[valid_cols].dropna()
        valid_indices = df_clean.index
        y_clean = y_raw[valid_indices]

        x_data = df_clean.to_numpy(dtype=np.float64)

        if spec.task == "classification":
            y_data = _binarize_target(y_clean, positive_class_label=spec.positive_class_label)
        else:
            y_data = np.asarray(y_clean, dtype=np.float64)

        interim_dir.mkdir(parents=True, exist_ok=True)
        np.save(x_path, x_data)
        np.save(y_path, y_data)

        logger.info(
            "Preprocessed and saved dataset '%s' to interim cache: X shape %s, y shape %s",
            spec.name,
            x_data.shape,
            y_data.shape,
        )
    except Exception as exc:
        msg = f"Failed to parse or preprocess dataset '{spec.name}' from raw cache."
        raise DatasetLoadError(msg) from exc

    return RawDataset(
        x=x_data,
        y=y_data,
        feature_names=tuple(valid_cols),
        source=f"interim_cache:{spec.name}",
        notes=f"Loaded from interim numpy cache for '{spec.name}'.",
    )


def _binarize_target(target: object, *, positive_class_label: int | str | None) -> IntArray:
    """Convert binary target labels into a 0/1 integer array.

    Args:
        target (object): Target values.
        positive_class_label (int | str | None): Value representing the positive class.

    Returns:
        IntArray: Binarized target values.

    Raises:
        DatasetLoadError: If positive_class_label is missing for multi-value targets.
    """
    target_array = np.asarray(target)
    if positive_class_label is None:
        unique = np.unique(target_array)
        if unique.size != 2:
            msg = "A positive_class_label is required for non-binary classification datasets."
            raise DatasetLoadError(msg)
        positive_class_label = unique[-1]
    string_target = np.asarray([str(value) for value in target_array], dtype=object)
    positive_label = str(positive_class_label)
    return np.asarray(string_target == positive_label, dtype=np.int64)


_LOADERS = {
    "openml": _load_openml,
}


def _load_yaml_datasets(key: str) -> list[DatasetSpec]:
    """Parse and load dataset specifications from YAML.

    Args:
        key (str): Registry block key in YAML.

    Returns:
        list[DatasetSpec]: List of DatasetSpec configurations.

    Raises:
        DatasetLoadError: If datasets.yaml is not found.
    """
    if not DATASETS_YAML_PATH.exists():
        msg = f"Centralized dataset configuration not found at {DATASETS_YAML_PATH}"
        raise DatasetLoadError(msg)
    with DATASETS_YAML_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return [
        DatasetSpec(
            name=entry["name"],
            task=entry["task"],
            dimension_bucket=entry["dimension_bucket"],
            n_features=entry["n_features"],
            source=entry["source"],
            description=entry.get("description", ""),
            url=entry.get("url"),
            model_family=entry.get("model_family"),
            builder_key=entry.get("builder_key"),
            loader_key=entry.get("loader_key"),
            positive_class_label=entry.get("positive_class_label"),
            openml_candidates=tuple(entry.get("openml_candidates", ())),
            openml_id=entry.get("openml_id"),
            target_column=entry.get("target_column"),
            is_multiclass=entry.get("is_multiclass", False),
        )
        for entry in data.get(key, [])
    ]


def datasets() -> list[DatasetSpec]:
    """Return the curated 12-dataset registry from datasets.yaml.

    Returns:
        list[DatasetSpec]: Curated list of DatasetSpec metadata.
    """
    specs = _load_yaml_datasets("datasets")
    validate_dataset_registry(specs)
    return specs


def default_dataset_registry() -> list[DatasetSpec]:
    """Return the combined registry for all experiments.

    Returns:
        list[DatasetSpec]: Registry of DatasetSpec metadata.
    """
    return datasets()


def build_raw_dataset(
    spec: DatasetSpec,
    _seed: int = 42,
) -> RawDataset:
    """Materialize and validate one real dataset.

    Args:
        spec (DatasetSpec): Metadata definition of the dataset.
        _seed (int): Random state seed.

    Returns:
        RawDataset: A RawDataset instance containing numeric features.

    Raises:
        DatasetLoadError: If loader fails or output feature dimensions don't match.
    """
    loader = _LOADERS.get(spec.loader_key) if spec.loader_key else None
    if not loader:
        msg = f"Unknown loader_key '{spec.loader_key}' for dataset '{spec.name}'."
        raise DatasetLoadError(msg)

    try:
        dataset = loader(spec)
    except Exception as exc:
        msg = f"Failed to load dataset '{spec.name}'."
        raise DatasetLoadError(msg) from exc

    _validate_numeric_feature_matrix(np.asarray(dataset.x, dtype=float), dataset_name=spec.name)

    if dataset.x.shape[1] != spec.n_features:
        msg = f"Dataset '{spec.name}' produced {dataset.x.shape[1]} features, expected {spec.n_features}."
        raise DatasetLoadError(msg)

    if spec.task == "classification":
        unique_targets = np.unique(dataset.y)
        if len(unique_targets) != 2:
            msg = f"Dataset '{spec.name}' is classification but has {len(unique_targets)} classes; only binary classification is supported."
            raise DatasetLoadError(msg)

    return dataset


def validate_dataset_registry(specs: list[DatasetSpec]) -> None:
    """Assert that the benchmark registry matches the design (12 datasets).

    Validates that there are exactly 2 datasets in each of the 6 blocks
    (regression/classification crossed with low/medium/high dimension buckets).

    Args:
        specs (list[DatasetSpec]): List of DatasetSpec configurations.

    Raises:
        ValueError: If the registry size or block partitioning is invalid.
    """
    if len(specs) != 12:
        msg = f"Expected 12 datasets, got {len(specs)}."
        raise ValueError(msg)

    counts: dict[tuple[TaskType, DimensionBucket], int] = {}
    for spec in specs:
        key = (spec.task, spec.dimension_bucket)
        counts[key] = counts.get(key, 0) + 1

    expected = {
        ("regression", "low"): 2,
        ("regression", "medium"): 2,
        ("regression", "high"): 2,
        ("classification", "low"): 2,
        ("classification", "medium"): 2,
        ("classification", "high"): 2,
    }
    for (task, bucket), expected_count in expected.items():
        if counts.get((task, bucket), 0) != expected_count:
            msg = f"Expected {expected_count} datasets for task={task}, bucket={bucket}; got {counts.get((task, bucket), 0)}."
            raise ValueError(msg)


if __name__ == "__main__":
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "datasets.log"

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[file_handler, stream_handler]
    )

    logger.info("Downloading and validating all registered datasets")
    try:
        specs = datasets()
        for idx, spec in enumerate(specs, 1):
            logger.info(
                "[%2d/12] Processing: %-25s (Task: %-15s Bucket: %-8s)",
                idx,
                spec.name,
                spec.task,
                spec.dimension_bucket,
            )
            # Materialize raw dataset to trigger download, caching, and validation checks
            build_raw_dataset(spec, _seed=42)
        logger.info("All 12 datasets downloaded, cached, and validated.")
    except (DatasetLoadError, ValueError, OSError) as e:
        logger.error("Error validating datasets: %s", e)
        sys.exit(1)
