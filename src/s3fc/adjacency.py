"""
S3FC Adjacency Module
=====================

Convert sparse coefficient matrix to adjacency matrix with K-nearest neighbor sparsification.
"""

import numpy as np
from scipy import sparse


def coeffs_to_adjacency(C, K=10, use_sparse=True, eps=1e-10, zero_diag=True, normalize_cols=True, verbose=False):
    """
    Convert coefficient matrix to (sparse) adjacency matrix.

    Parameters
    ----------
    C : ndarray of shape (n_samples, n_samples)
        Sparse coefficient matrix from subspace clustering.
    K : int, default=10
        Number of nearest neighbors for sparsification.
        Set K=0 for full (dense) adjacency.
    use_sparse : bool, default=True
        If True, return scipy.sparse matrix (O(N*K) memory).
        If False, return dense numpy array (O(N^2) memory).
    eps : float, default=1e-10
        Small constant for numerical stability.
    zero_diag : bool, default=True
        If True, set diagonal to zero (no self-loops).
    normalize_cols : bool, default=True
        If True, normalize columns by their maximum value.
    verbose : bool, default=False
        If True, print memory statistics.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix or ndarray
        Symmetric adjacency matrix.
    memory_stats : dict
        Memory usage statistics (only if use_sparse=True and K>0).

    Notes
    -----
    Processing steps:
    1. Take absolute values of coefficients
    2. Normalize columns by max value (optional)
    3. Symmetrize: C_sym = C + C.T
    4. Keep only K nearest neighbors per sample (if K > 0)
    5. Symmetrize again after sparsification
    """
    N = C.shape[0]
    memory_stats = None

    # Handle sparse input from coefficients module
    if sparse.issparse(C):
        C = C.copy()
        C.data = np.abs(C.data)
        if zero_diag:
            C.setdiag(0)
            C.eliminate_zeros()
        if normalize_cols:
            # Normalize each column by its max value
            C = C.tocsc()
            for j in range(N):
                col_data = C.getcol(j).data
                if len(col_data) > 0:
                    cmax = col_data.max() + eps
                    start, end = C.indptr[j], C.indptr[j + 1]
                    C.data[start:end] /= cmax
        # Symmetrize
        symC = C + C.T
        if zero_diag:
            symC.setdiag(0)
            symC.eliminate_zeros()
    else:
        C = np.abs(C)
        if zero_diag:
            np.fill_diagonal(C, 0.0)
        if normalize_cols:
            col_max = np.max(C, axis=0) + eps
            C = C / col_max
        symC = C + C.T
        if zero_diag:
            np.fill_diagonal(symC, 0.0)

    # K=0 means no sparsification (full adjacency)
    if K == 0:
        if sparse.issparse(symC):
            return symC.tocsr(), None
        return (sparse.csr_matrix(symC), None) if use_sparse else (symC, None)

    if use_sparse:
        # Build sparse K-nearest neighbor adjacency
        rows, cols, data = [], [], []

        # Convert to CSC for efficient column access
        if sparse.issparse(symC):
            symC_csc = symC.tocsc()
        else:
            symC_csc = None

        for i in range(N):
            if symC_csc is not None:
                # Sparse: extract column as dense for argsort
                col = symC_csc.getcol(i).toarray().ravel()
            else:
                col = symC[:, i]
            # Get indices of K largest values
            idx = np.argsort(-col)[:K]
            denom = col[idx[0]] + eps

            for j in idx:
                if zero_diag and j == i:
                    continue
                rows.append(j)
                cols.append(i)
                data.append(col[j] / denom)

        # Create sparse matrix and symmetrize
        topK = sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        topKsym = topK + topK.T

        if zero_diag:
            topKsym.setdiag(0)
            topKsym.eliminate_zeros()

        # Compute memory statistics
        memory_stats = {
            'adj_nnz': topKsym.nnz,
            'adj_sparse_mb': (topKsym.data.nbytes + topKsym.indices.nbytes +
                              topKsym.indptr.nbytes) / 1e6,
            'adj_dense_mb': N * N * 8 / 1e6,
        }

        if verbose:
            savings = memory_stats['adj_dense_mb'] / max(memory_stats['adj_sparse_mb'], 0.001)
            print(f"Memory: {memory_stats['adj_sparse_mb']:.2f} MB sparse vs "
                  f"{memory_stats['adj_dense_mb']:.2f} MB dense ({savings:.1f}x savings)")

        return topKsym, memory_stats
    else:
        # Dense K-nearest neighbor adjacency
        if sparse.issparse(symC):
            symC = symC.toarray()
        topK = np.zeros((N, N), dtype=float)
        for i in range(N):
            col = symC[:, i]
            idx = np.argsort(-col)[:K]
            denom = col[idx[0]] + eps
            topK[idx, i] = col[idx] / denom

        topKsym = topK + topK.T
        if zero_diag:
            np.fill_diagonal(topKsym, 0.0)

        return topKsym, None
