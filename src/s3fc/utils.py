"""Common utility functions for s3fc."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_data(
    X: NDArray[np.floating],
    y: NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64] | None]:
    """Validate and convert data to standard format.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,), optional
        Ground truth labels.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features), dtype float64
        Validated feature matrix.
    y : ndarray of shape (n_samples,), dtype int64, or None
        Validated labels (if provided).

    Raises
    ------
    ValueError
        If X is not 2D, y is not 1D, or shapes don't match.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")

    if y is not None:
        y = np.asarray(y, dtype=np.int64)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got {y.ndim}D")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"{X.shape[0]} vs {len(y)}"
            )

    return X, y


def check_random_state(seed: int | np.random.RandomState | None) -> np.random.RandomState:
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : int, RandomState, or None
        If int, use as seed for new RandomState.
        If RandomState, return it.
        If None, return the global random state.

    Returns
    -------
    np.random.RandomState
        A RandomState instance.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"Cannot convert {seed!r} to a RandomState instance")


def relabel_sequential(labels: NDArray[np.integer]) -> NDArray[np.int64]:
    """Relabel cluster labels to sequential integers starting from 0.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Cluster labels (can be non-sequential or negative).

    Returns
    -------
    ndarray of shape (n_samples,), dtype int64
        Labels remapped to 0, 1, 2, ...
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map[label] for label in labels], dtype=np.int64)


def count_clusters(labels: NDArray[np.integer]) -> int:
    """Count the number of unique clusters in labels.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Cluster labels.

    Returns
    -------
    int
        Number of unique clusters.
    """
    return len(np.unique(labels))
