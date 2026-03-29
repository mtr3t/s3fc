"""
S3FC Coefficients Module
========================

Sparse coefficient optimization using CVXPY for subspace clustering.
"""

import numpy as np
import cvxpy as cp
from scipy import sparse
from scipy.spatial import KDTree
from joblib import Parallel, delayed


def solve_single_coefficient(i, Y, N, gamma, affine=False, solver='ECOS',
                             neighbor_idx=None):
    """
    Solve sparse coefficient optimization for a single sample.

    Solves: min_x  gamma * ||Y_{-i} @ x - y_i||_2 + ||x||_1
            s.t.   sum(x) == 1  (if affine=True)

    Parameters
    ----------
    i : int
        Index of the current sample.
    Y : ndarray of shape (n_features, n_samples)
        Data matrix (transposed, features x samples).
    N : int
        Total number of samples.
    gamma : float
        Regularization weight balancing reconstruction vs sparsity.
    affine : bool, default=False
        If True, enforce affine constraint (coefficients sum to 1).
    solver : str, default='ECOS'
        CVXPY solver to use. Options: 'ECOS', 'SCS', 'OSQP'.
    neighbor_idx : ndarray or None
        If provided, indices of dictionary atoms to use (K_dict nearest
        neighbors). Reduces problem size from N-1 to len(neighbor_idx).

    Returns
    -------
    i : int
        Index of the sample (for parallel assembly).
    x_val : ndarray of shape (N-1,) or tuple (indices, values) if neighbor_idx
        Solved coefficient values.
    status : str
        Solver status ('optimal', 'failed', etc.).

    Notes
    -----
    The optimization encourages sparse representation of sample i
    using all other samples as a dictionary. This is the core of
    sparse subspace clustering (SSC).
    """
    b = Y[:, i]

    if neighbor_idx is not None:
        # Restricted dictionary: only use K_dict nearest neighbors
        idx = neighbor_idx[neighbor_idx != i]  # exclude self
        Y_dict = Y[:, idx]
        n_dict = len(idx)

        x = cp.Variable(n_dict)
        obj = cp.Minimize(
            gamma * cp.norm(Y_dict @ x - b, 2) + cp.norm1(x)
        )
        constraints = [cp.sum(x) == 1] if affine else []
        prob = cp.Problem(obj, constraints)

        try:
            prob.solve(solver=solver)
            x_val = x.value if x.value is not None else np.zeros(n_dict)
            status = prob.status
        except Exception:
            x_val = np.zeros(n_dict)
            status = "failed"

        # Return (indices, values) tuple for sparse assembly
        return i, (idx, x_val), status
    else:
        # Full dictionary: all N-1 other samples
        Y_minus_i = np.delete(Y, i, axis=1)

        x = cp.Variable(N - 1)
        obj = cp.Minimize(
            gamma * cp.norm(Y_minus_i @ x - b, 2) + cp.norm1(x)
        )
        constraints = [cp.sum(x) == 1] if affine else []
        prob = cp.Problem(obj, constraints)

        try:
            prob.solve(solver=solver)
            x_val = x.value if x.value is not None else np.zeros(N - 1)
            status = prob.status
        except Exception:
            x_val = np.zeros(N - 1)
            status = "failed"

        return i, x_val, status


def sparse_coefficients(X, gamma, affine=False, solver='ECOS', n_jobs=1,
                        verbose=False, K_dict=None):
    """
    Compute sparse coefficient matrix for all samples.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix.
    gamma : float
        Regularization weight for sparse optimization.
    affine : bool, default=False
        If True, enforce affine constraint.
    solver : str, default='ECOS'
        CVXPY solver to use.
    n_jobs : int, default=1
        Number of parallel jobs. -1 means use all cores.
    verbose : bool, default=False
        If True, print progress information.
    K_dict : int or None, default=None
        If set, restrict each CVXPY problem's dictionary to the K_dict
        nearest neighbors. Reduces per-problem variables from N-1 to K_dict,
        giving massive speedup for large N. The L1 sparsity penalty drives
        distant points' coefficients to zero anyway, so this does not change
        the solution in practice.

    Returns
    -------
    C : scipy.sparse.csc_matrix of shape (n_samples, n_samples)
        Sparse coefficient matrix where C[j,i] is the weight of
        sample j in representing sample i. Stored in sparse format
        to support large N (O(nnz) memory instead of O(N^2)).

    Notes
    -----
    This function is embarrassingly parallel - each sample's coefficients
    can be computed independently. Use n_jobs=-1 for maximum parallelism.
    """
    Y = X.T  # Features x Samples
    _, N = Y.shape

    # Precompute nearest neighbors if K_dict is set
    nn_indices = None
    if K_dict is not None and K_dict < N - 1:
        if verbose:
            print(f"Building KD-tree for K_dict={K_dict} nearest neighbors...")
        tree = KDTree(X)
        # Query K_dict+1 because the point itself is included
        _, nn_indices = tree.query(X, k=K_dict + 1)
        if verbose:
            print(f"Solving {N} sparse coefficient problems "
                  f"(K_dict={K_dict}, {K_dict} variables each)...")
    else:
        K_dict = None  # disable if K_dict >= N-1
        if verbose:
            print(f"Solving {N} sparse coefficient problems "
                  f"({N-1} variables each)...")

    # Sequential or parallel execution
    if n_jobs == 1:
        results = [
            solve_single_coefficient(
                i, Y, N, gamma, affine, solver,
                neighbor_idx=nn_indices[i] if nn_indices is not None else None
            )
            for i in range(N)
        ]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(solve_single_coefficient)(
                i, Y, N, gamma, affine, solver,
                neighbor_idx=nn_indices[i] if nn_indices is not None else None
            )
            for i in range(N)
        )

    # Assemble directly into sparse COO format (O(nnz) memory)
    rows, cols, data = [], [], []
    for i, x_val, _ in results:
        if x_val is None:
            continue

        if K_dict is not None:
            # x_val is (indices, values) tuple from restricted dictionary
            idx, vals = x_val
            if vals is None:
                continue
            nz = np.nonzero(vals)[0]
            for pos in nz:
                rows.append(idx[pos])
                cols.append(i)
                data.append(vals[pos])
        else:
            # x_val has N-1 entries (diagonal skipped)
            # Map back to full indices: positions 0..i-1 -> rows 0..i-1,
            #                           positions i..N-2 -> rows i+1..N-1
            nz = np.nonzero(x_val)[0]
            for pos in nz:
                row = pos if pos < i else pos + 1
                rows.append(row)
                cols.append(i)
                data.append(x_val[pos])

    C = sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsc()

    if verbose:
        density = C.nnz / (N * N) * 100
        mb = (C.data.nbytes + C.indices.nbytes + C.indptr.nbytes) / 1e6
        dense_mb = N * N * 8 / 1e6
        print(f"Coefficient matrix: {C.nnz} nonzeros ({density:.1f}% dense), "
              f"{mb:.1f} MB sparse vs {dense_mb:.1f} MB dense")

    return C
