"""
S3FC Affinity Module
====================

RBF (Radial Basis Function) affinity computation with both dense and sparse modes.
"""

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def dense_rbf_affinity(X, sigma, eps=1e-10, zero_diag=True):
    """
    Compute dense RBF affinity matrix - O(N^2) time and space.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    sigma : float
        RBF kernel bandwidth parameter.
    eps : float, default=1e-10
        Small constant for numerical stability.
    zero_diag : bool, default=True
        If True, set diagonal to zero (no self-affinity).

    Returns
    -------
    A : ndarray of shape (n_samples, n_samples)
        Dense RBF affinity matrix.

    Notes
    -----
    The RBF kernel is: A[i,j] = exp(-||x_i - x_j||^2 / (2 * sigma^2))

    This is the original O(N^2) implementation. For large N, consider
    using sparse_rbf_affinity() instead.
    """
    # Compute pairwise squared distances via broadcasting
    # diff[i,j,d] = X[i,d] - X[j,d]
    diff = X[:, None, :] - X[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)

    # Apply RBF kernel
    A = np.exp(-dist2 / (2.0 * sigma ** 2 + eps))

    if zero_diag:
        np.fill_diagonal(A, 0.0)

    return A


def sparse_rbf_affinity(X, sigma, K_rbf=30, eps=1e-10, zero_diag=True, verbose=False):
    """
    Compute sparse RBF affinity using K nearest neighbors - O(N*K) time and space.

    Only computes RBF values for K_rbf nearest neighbors per point,
    storing the result as a sparse matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    sigma : float
        RBF kernel bandwidth parameter.
    K_rbf : int, default=30
        Number of nearest neighbors to consider per point.
    eps : float, default=1e-10
        Small constant for numerical stability.
    zero_diag : bool, default=True
        If True, exclude self-connections.
    verbose : bool, default=False
        If True, print progress information.

    Returns
    -------
    A_sparse : scipy.sparse.csr_matrix of shape (n_samples, n_samples)
        Sparse RBF affinity matrix.

    Notes
    -----
    This reduces memory from O(N^2) to O(N*K_rbf), enabling scaling to
    large datasets. The matrix is symmetrized: A = (A + A.T) / 2.
    """
    N = X.shape[0]
    K = min(K_rbf, N - 1)  # Can't have more neighbors than points

    if verbose:
        print(f"Computing sparse RBF with K={K} neighbors...")

    # Find K nearest neighbors using sklearn
    nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Build sparse matrix data
    rows = []
    cols = []
    data = []

    for i in range(N):
        for j_idx, j in enumerate(indices[i]):
            if zero_diag and i == j:
                continue
            dist2 = distances[i, j_idx] ** 2
            rbf_val = np.exp(-dist2 / (2.0 * sigma ** 2 + eps))
            rows.append(i)
            cols.append(j)
            data.append(rbf_val)

    # Create sparse CSR matrix
    A_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    # Symmetrize: A = (A + A.T) / 2
    A_sparse = (A_sparse + A_sparse.T) / 2

    return A_sparse


def self_tuning_affinity(X, K_local=7, eps=1e-10, zero_diag=True, verbose=False):
    """
    Compute self-tuning affinity matrix (Zelnik-Manor & Perona, 2004).

    Uses local scaling where sigma_i = distance to K_local-th neighbor.
    A[i,j] = exp(-d(i,j)^2 / (sigma_i * sigma_j))

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    K_local : int, default=7
        Which neighbor's distance to use as local scale.
        Paper recommends K=7.
    eps : float, default=1e-10
        Small constant for numerical stability.
    zero_diag : bool, default=True
        If True, set diagonal to zero.
    verbose : bool, default=False
        If True, print progress information.

    Returns
    -------
    A : ndarray of shape (n_samples, n_samples)
        Self-tuning affinity matrix.

    Notes
    -----
    This method adapts the kernel bandwidth locally based on data density.
    Dense regions get smaller sigma, sparse regions get larger sigma.
    Removes the need to tune a global sigma parameter.

    Reference: Zelnik-Manor & Perona, "Self-Tuning Spectral Clustering", NIPS 2004.
    """
    N = X.shape[0]
    K = min(K_local, N - 1)

    if verbose:
        print(f"Computing self-tuning affinity with K_local={K}...")

    # Find K nearest neighbors
    nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto')
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # Local scale: distance to K-th neighbor (index K since index 0 is self)
    sigma_local = distances[:, K] + eps

    # Compute pairwise squared distances
    diff = X[:, None, :] - X[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)

    # Self-tuning kernel: exp(-d^2 / (sigma_i * sigma_j))
    sigma_product = np.outer(sigma_local, sigma_local)
    A = np.exp(-dist2 / (sigma_product + eps))

    if zero_diag:
        np.fill_diagonal(A, 0.0)

    return A


def rbf_affinity(X, sigma, sparse_rbf=False, K_rbf=30, eps=1e-10, zero_diag=True, verbose=False):
    """
    Compute RBF affinity matrix (dispatcher for dense/sparse modes).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    sigma : float
        RBF kernel bandwidth parameter.
    sparse_rbf : bool, default=False
        If True, use sparse RBF via K nearest neighbors.
        If False, use dense O(N^2) computation.
    K_rbf : int, default=30
        Number of neighbors for sparse RBF (only used if sparse_rbf=True).
    eps : float, default=1e-10
        Small constant for numerical stability.
    zero_diag : bool, default=True
        If True, set diagonal to zero.
    verbose : bool, default=False
        If True, print progress information.

    Returns
    -------
    A : ndarray or scipy.sparse.csr_matrix
        RBF affinity matrix (dense or sparse depending on sparse_rbf).
    """
    if sparse_rbf:
        return sparse_rbf_affinity(X, sigma, K_rbf, eps, zero_diag, verbose)
    else:
        return dense_rbf_affinity(X, sigma, eps, zero_diag)
