"""Evaluation metrics for clustering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def nmi_score(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    average_method: str = "arithmetic",
) -> float:
    """Compute Normalized Mutual Information between two clusterings.

    Wrapper around sklearn.metrics.normalized_mutual_info_score with
    default average_method='arithmetic' (standard choice for clustering).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth cluster labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.
    average_method : str, default='arithmetic'
        How to compute the normalizer in the denominator.
        Options: 'min', 'geometric', 'arithmetic', 'max'.

    Returns
    -------
    float
        NMI score in [0, 1]. 1 means perfect agreement.

    Examples
    --------
    >>> y_true = [0, 0, 1, 1]
    >>> y_pred = [0, 0, 1, 1]
    >>> nmi_score(y_true, y_pred)
    1.0

    >>> y_pred = [1, 1, 0, 0]  # Same clustering, different labels
    >>> nmi_score(y_true, y_pred)
    1.0
    """
    return float(
        normalized_mutual_info_score(
            y_true, y_pred, average_method=average_method
        )
    )


def ari_score(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
) -> float:
    """Compute Adjusted Rand Index between two clusterings.

    Wrapper around sklearn.metrics.adjusted_rand_score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth cluster labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    float
        ARI score in [-1, 1]. 1 means perfect agreement, 0 means random.

    Examples
    --------
    >>> y_true = [0, 0, 1, 1]
    >>> y_pred = [0, 0, 1, 1]
    >>> ari_score(y_true, y_pred)
    1.0
    """
    return float(adjusted_rand_score(y_true, y_pred))


def clustering_accuracy(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
) -> float:
    """Compute clustering accuracy using Hungarian algorithm.

    Finds the optimal label mapping between predicted and true labels
    to maximize accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth cluster labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    float
        Accuracy in [0, 1]. 1 means perfect match after optimal relabeling.

    Examples
    --------
    >>> y_true = [0, 0, 1, 1]
    >>> y_pred = [1, 1, 0, 0]  # Flipped labels
    >>> clustering_accuracy(y_true, y_pred)
    1.0
    """
    from scipy.optimize import linear_sum_assignment

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_samples = len(y_true)
    n_true_clusters = len(np.unique(y_true))
    n_pred_clusters = len(np.unique(y_pred))
    n_clusters = max(n_true_clusters, n_pred_clusters)

    # Build cost matrix (negative overlap counts)
    cost_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i, true_label in enumerate(np.unique(y_true)):
        for j, pred_label in enumerate(np.unique(y_pred)):
            cost_matrix[i, j] = -np.sum((y_true == true_label) & (y_pred == pred_label))

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    accuracy = -cost_matrix[row_ind, col_ind].sum() / n_samples

    return float(accuracy)
