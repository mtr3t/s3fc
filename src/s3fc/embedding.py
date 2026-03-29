"""
S3FC Embedding Module
=====================

Spectral embedding via normalized Laplacian eigendecomposition.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def spectral_embedding(A, n_clusters, eps=1e-10):
    """
    Compute spectral embedding from affinity matrix.

    Uses the normalized Laplacian approach:
    L_sym = D^{-1/2} @ A @ D^{-1/2}

    Parameters
    ----------
    A : ndarray or scipy.sparse matrix of shape (n_samples, n_samples)
        Symmetric affinity matrix.
    n_clusters : int
        Number of clusters (determines embedding dimension).
    eps : float, default=1e-10
        Small constant for numerical stability.

    Returns
    -------
    U : ndarray of shape (n_samples, n_clusters)
        Row-normalized spectral embedding.

    Notes
    -----
    The embedding consists of the top n_clusters eigenvectors of the
    normalized Laplacian, with rows normalized to unit length.

    For sparse matrices, uses scipy.sparse.linalg.eigsh (ARPACK).
    For dense matrices, uses numpy.linalg.eigh.
    """
    is_sparse = sparse.issparse(A)

    if is_sparse:
        # Sparse path: use eigsh (ARPACK) for efficiency
        d = np.asarray(A.sum(axis=1)).flatten()
        d_inv_sqrt = 1.0 / np.sqrt(d + eps)
        D_inv_sqrt = sparse.diags(d_inv_sqrt)

        # Normalized Laplacian (with small regularization for stability)
        L = D_inv_sqrt @ A @ D_inv_sqrt + sparse.eye(A.shape[0]) * eps

        # Get top k eigenvectors (largest eigenvalues)
        _, eigvecs = eigsh(L, k=n_clusters, which='LM')
        U = eigvecs
    else:
        # Dense path: use full eigendecomposition
        d = np.sum(A, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d + eps))

        # Normalized Laplacian
        L = D_inv_sqrt @ A @ D_inv_sqrt + eps

        # Full eigendecomposition, take last k (largest) eigenvectors
        _, eigvecs = np.linalg.eigh(L)
        U = eigvecs[:, -n_clusters:]

    # Row-normalize the embedding
    row_norms = np.linalg.norm(U, axis=1, keepdims=True) + eps
    return U / row_norms
