"""Data loading utilities for s3fc.

Supports:
- sklearn toy datasets (digits, iris, wine, etc.)
- OpenML datasets (mnist_784, Fashion-MNIST, usps, etc.)
- Legacy .tp/.ds formats from original notebooks
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .utils import validate_data

# Mapping of common dataset names to sklearn loaders
SKLEARN_DATASETS = {
    "digits": "load_digits",
    "iris": "load_iris",
    "wine": "load_wine",
    "breast_cancer": "load_breast_cancer",
}

# Mapping of common names to OpenML dataset IDs
OPENML_DATASETS = {
    "mnist_784": 554,
    "fashion-mnist": 40996,
    "usps": 41082,
    "pendigits": 32,
    "letter": 6,
    "coil20": 41971,
    "har": 1478,  # Human Activity Recognition
}


def load_dataset(
    name: str,
    return_X_y: bool = True,
    cache: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load a standard dataset by name.

    Tries sklearn first, then OpenML.

    Parameters
    ----------
    name : str
        Dataset name. Supported:
        - sklearn: 'digits', 'iris', 'wine', 'breast_cancer'
        - OpenML: 'mnist_784', 'fashion-mnist', 'usps', 'pendigits', 'letter'
    return_X_y : bool, default=True
        If True, return (X, y) tuple. If False, return Bunch object.
    cache : bool, default=True
        Whether to cache OpenML downloads.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Ground truth labels.

    Raises
    ------
    ValueError
        If dataset name is not recognized.

    Examples
    --------
    >>> X, y = load_dataset("digits")
    >>> X.shape
    (1797, 64)
    >>> y.shape
    (1797,)
    """
    name_lower = name.lower()

    # Try sklearn first
    if name_lower in SKLEARN_DATASETS:
        return _load_sklearn(name_lower)

    # Try OpenML
    if name_lower in OPENML_DATASETS:
        return _load_openml(OPENML_DATASETS[name_lower], cache=cache)

    # Try as raw OpenML ID
    if name.isdigit():
        return _load_openml(int(name), cache=cache)

    raise ValueError(
        f"Unknown dataset: {name}. "
        f"Supported sklearn: {list(SKLEARN_DATASETS.keys())}. "
        f"Supported OpenML: {list(OPENML_DATASETS.keys())}. "
        f"Or provide an OpenML dataset ID as a string."
    )


def _load_sklearn(name: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load a sklearn dataset."""
    from sklearn import datasets

    loader = getattr(datasets, SKLEARN_DATASETS[name])
    data = loader()
    X, y = validate_data(data.data, data.target)
    return X, y


def _load_openml(
    data_id: int,
    cache: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load a dataset from OpenML."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml(
        data_id=data_id,
        as_frame=False,
        cache=cache,
        parser="auto",
    )
    X, y = validate_data(data.data, data.target.astype(int))
    return X, y


def load_legacy(
    data_path: str | Path,
    labels_path: str | Path | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load data from legacy .tp or .ds format.

    Legacy formats:
    - .tp files: space-delimited features per line
    - .ds files: space-delimited features per line (e.g., 320 values for 20x16 images)
    - ground truth: one integer label per line

    Parameters
    ----------
    data_path : str or Path
        Path to the data file (.tp or .ds).
    labels_path : str or Path, optional
        Path to ground truth labels file.
        If None, tries to find *_ground_truth.* or ground_truth.* in same directory.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Ground truth labels.

    Raises
    ------
    FileNotFoundError
        If data file or labels file not found.
    ValueError
        If data format is invalid.

    Examples
    --------
    >>> X, y = load_legacy("data.tp", "labels.tp")
    >>> X, y = load_legacy("data.ds")  # Auto-finds ground truth
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load features
    X = _load_legacy_features(data_path)

    # Find and load labels
    if labels_path is None:
        labels_path = _find_ground_truth(data_path)
    else:
        labels_path = Path(labels_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    y = _load_legacy_labels(labels_path)

    X, y = validate_data(X, y)
    return X, y


def _load_legacy_features(path: Path) -> NDArray[np.float64]:
    """Load features from a legacy file."""
    lines = path.read_text().strip().split("\n")
    data = []
    for line in lines:
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        data.append(values)
    return np.array(data, dtype=np.float64)


def _load_legacy_labels(path: Path) -> NDArray[np.int64]:
    """Load labels from a legacy file."""
    lines = path.read_text().strip().split("\n")
    labels = []
    for line in lines:
        if not line.strip():
            continue
        labels.append(int(float(line.strip())))
    return np.array(labels, dtype=np.int64)


def _find_ground_truth(data_path: Path) -> Path:
    """Try to find ground truth file for a data file.

    Search patterns:
    1. {stem}_ground_truth.{ext}
    2. ground_truth.{ext}
    3. {stem}_ground_truth.*
    4. ground_truth.*
    """
    parent = data_path.parent
    stem = data_path.stem
    ext = data_path.suffix

    # Pattern 1: same_name_ground_truth.same_ext
    candidate = parent / f"{stem}_ground_truth{ext}"
    if candidate.exists():
        return candidate

    # Pattern 2: ground_truth.same_ext
    candidate = parent / f"ground_truth{ext}"
    if candidate.exists():
        return candidate

    # Pattern 3: same_name_ground_truth.*
    candidates = list(parent.glob(f"{stem}_ground_truth.*"))
    if candidates:
        return candidates[0]

    # Pattern 4: ground_truth.*
    candidates = list(parent.glob("ground_truth.*"))
    if candidates:
        return candidates[0]

    # Pattern 5: *_ground_truth.* (any file with ground_truth in name)
    candidates = list(parent.glob("*ground_truth*"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find ground truth file for {data_path}. "
        f"Tried patterns: {stem}_ground_truth{ext}, ground_truth{ext}, "
        f"{stem}_ground_truth.*, ground_truth.*"
    )


def list_available_datasets() -> dict[str, list[str]]:
    """List all available datasets by source.

    Returns
    -------
    dict
        Dictionary with keys 'sklearn', 'openml' containing lists of dataset names.
    """
    return {
        "sklearn": list(SKLEARN_DATASETS.keys()),
        "openml": list(OPENML_DATASETS.keys()),
    }
