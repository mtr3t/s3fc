"""
S3FC Core Implementation
========================

Main clustering class implementing Sparse Spectral Subspace Fusion Clustering.
This module orchestrates the algorithm flow using sub-modules.
"""

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin

from .affinity import rbf_affinity, self_tuning_affinity
from .coefficients import sparse_coefficients
from .adjacency import coeffs_to_adjacency
from .embedding import spectral_embedding


class S3FC(BaseEstimator, ClusterMixin):
    """
    Scalable Sparse Spectral Fusion Clustering.

    Fuses sparse subspace affinity with spectral (RBF) affinity for
    multi-manifold clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.

    gamma : float, default=1.0
        Regularization weight for sparse coefficient optimization.
        Higher values encourage sparser solutions.

    affinity_type : {'rbf', 'self_tuning'}, default='rbf'
        Type of spectral affinity to use:
        - 'rbf': Standard RBF/Gaussian kernel (requires sigma parameter)
        - 'self_tuning': Zelnik-Manor & Perona local scaling (adapts sigma per-point)

    sigma : float, default=16.0
        RBF kernel bandwidth. Controls the scale of the spectral affinity.
        Only used when affinity_type='rbf'.

    K_local : int, default=7
        Number of neighbors for local scaling in self-tuning affinity.
        Only used when affinity_type='self_tuning'. Paper recommends K=7.

    K : int, default=10
        Number of nearest neighbors for adjacency sparsification.
        Set K=0 for full (dense) adjacency.

    fusion : {'jordan', 'legacy', 'hadamard', 'sum', 'power', 'geometric'}, default='jordan'
        Fusion strategy for combining affinities:
        - 'jordan': 0.5*(A_sp @ Adj + Adj @ A_sp) - best for real data
        - 'legacy': A_sp @ Adj - best for synthetic manifolds
        - 'hadamard': A_sp * Adj - element-wise product
        - 'sum': alpha*A_sp + (1-alpha)*Adj - weighted sum (use fusion_alpha)
        - 'power': (A_sp @ Adj)^k - multiple diffusion steps (use fusion_power)
        - 'geometric': sqrt(A_sp * Adj) - element-wise geometric mean

    affine : bool, default=False
        If True, enforce affine constraint (coefficients sum to 1).

    use_sparse : bool, default=True
        If True, use scipy.sparse for adjacency matrix (O(N*K) memory).
        If False, use dense numpy array (O(N^2) memory).

    sparse_rbf : bool, default=False
        If True, use sparse RBF affinity via K nearest neighbors.
        Reduces RBF computation from O(N^2) to O(N*K_rbf).

    K_rbf : int, default=30
        Number of neighbors for sparse RBF. Only used if sparse_rbf=True.

    K_dict : int or None, default=None
        If set, restrict each CVXPY problem's dictionary to the K_dict
        nearest neighbors. Reduces per-problem complexity from O(N) to
        O(K_dict) variables, enabling scalability to large N. The L1
        penalty drives distant coefficients to zero, so restricting
        the dictionary preserves solution quality.

    n_jobs : int, default=1
        Number of parallel jobs for coefficient solving.
        -1 means use all available cores.

    solver : str, default='ECOS'
        CVXPY solver for coefficient optimization.
        Options: 'ECOS', 'SCS', 'OSQP'.

    verbose : bool, default=False
        If True, print progress information.

    fusion_alpha : float, default=0.5
        Weight for 'sum' fusion: alpha*A_sp + (1-alpha)*Adj.
        Only used when fusion='sum'.

    fusion_power : int, default=2
        Number of diffusion steps for 'power' fusion: (A_sp @ Adj)^k.
        Only used when fusion='power'.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each sample.

    embedding_ : ndarray of shape (n_samples, n_clusters)
        Spectral embedding used for clustering.

    memory_stats_ : dict
        Memory usage statistics (sparse vs dense).

    Examples
    --------
    >>> from s3fc import S3FC
    >>> from sklearn.datasets import load_digits
    >>> X, y = load_digits(return_X_y=True)
    >>> model = S3FC(n_clusters=10, fusion='jordan', n_jobs=-1)
    >>> labels = model.fit_predict(X)
    >>> from sklearn.metrics import normalized_mutual_info_score
    >>> print(f"NMI: {normalized_mutual_info_score(y, labels):.4f}")

    References
    ----------
    Builds on sparse subspace clustering (Elhamifar & Vidal, 2013)
    and spectral clustering methods.
    """

    def __init__(
        self,
        n_clusters,
        gamma=1.0,
        affinity_type="rbf",
        sigma=16.0,
        K_local=7,
        K=10,
        fusion="jordan",
        affine=False,
        use_sparse=True,
        sparse_rbf=False,
        K_rbf=30,
        K_dict=None,
        n_jobs=1,
        solver="ECOS",
        verbose=False,
        random_state=None,
        fusion_alpha=0.5,
        fusion_power=2,
    ):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.affinity_type = affinity_type
        self.sigma = sigma
        self.K_local = K_local
        self.K = K
        self.fusion = fusion
        self.affine = affine
        self.use_sparse = use_sparse
        self.sparse_rbf = sparse_rbf
        self.K_rbf = K_rbf
        self.K_dict = K_dict
        self.n_jobs = n_jobs
        self.solver = solver
        self.verbose = verbose
        self.random_state = random_state
        self.fusion_alpha = fusion_alpha
        self.fusion_power = fusion_power

        # Internal parameters
        self._eps = 1e-10
        self._zero_diag = True
        self._normalize_cols = True
        self._symmetrize = False

    def _fuse_affinities(self, A_sp, Adj):
        """
        Fuse spectral (RBF) and subspace affinities.

        Parameters
        ----------
        A_sp : ndarray or sparse matrix
            Spectral affinity from RBF kernel.
        Adj : ndarray or sparse matrix
            Subspace adjacency from sparse coefficients.

        Returns
        -------
        A_fused : ndarray or sparse matrix
            Fused affinity matrix.
        """
        is_sparse_rbf = sparse.issparse(A_sp)
        is_sparse_adj = sparse.issparse(Adj)

        if self.fusion == "legacy":
            # A_sp @ Adj - best for synthetic manifolds
            A_fused = A_sp @ Adj

        elif self.fusion == "jordan":
            # 0.5*(A_sp @ Adj + Adj @ A_sp) - best for real data
            A_fused = 0.5 * (A_sp @ Adj + Adj @ A_sp)

        elif self.fusion == "hadamard":
            # Element-wise product - handle sparse matrices
            if is_sparse_rbf or is_sparse_adj:
                if not is_sparse_rbf:
                    A_sp = sparse.csr_matrix(A_sp)
                if not is_sparse_adj:
                    Adj = sparse.csr_matrix(Adj)
                A_fused = A_sp.multiply(Adj)
            else:
                A_fused = A_sp * Adj

        elif self.fusion == "sum":
            # Weighted sum: alpha*A_sp + (1-alpha)*Adj
            alpha = self.fusion_alpha
            # Handle mixed sparse/dense - convert to same type
            if is_sparse_rbf != is_sparse_adj:
                if is_sparse_adj:
                    Adj = Adj.toarray()
                else:
                    A_sp = A_sp if not is_sparse_rbf else A_sp.toarray()
            A_fused = alpha * A_sp + (1 - alpha) * Adj

        elif self.fusion == "power":
            # Multiple diffusion steps: (A_sp @ Adj)^k
            base = A_sp @ Adj
            A_fused = base
            for _ in range(self.fusion_power - 1):
                A_fused = A_fused @ base
            # Normalize to prevent explosion
            if sparse.issparse(A_fused):
                max_val = A_fused.max()
            else:
                max_val = np.max(A_fused)
            if max_val > 0:
                A_fused = A_fused / max_val

        elif self.fusion == "geometric":
            # Element-wise geometric mean: sqrt(A_sp * Adj)
            if is_sparse_rbf or is_sparse_adj:
                if not is_sparse_rbf:
                    A_sp = sparse.csr_matrix(A_sp)
                if not is_sparse_adj:
                    Adj = sparse.csr_matrix(Adj)
                A_fused = A_sp.multiply(Adj)
                A_fused.data = np.sqrt(A_fused.data)
            else:
                A_fused = np.sqrt(A_sp * Adj)

        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

        # Optional post-processing
        if self._symmetrize:
            A_fused = 0.5 * (A_fused + A_fused.T)

        if self._zero_diag:
            if sparse.issparse(A_fused):
                A_fused.setdiag(0)
                A_fused.eliminate_zeros()
            else:
                np.fill_diagonal(A_fused, 0.0)

        return A_fused

    def fit(self, X, y=None):
        """
        Fit S3FC model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)

        if self.verbose:
            print(f"S3FC: {X.shape[0]} samples, {X.shape[1]} features, "
                  f"{self.n_clusters} clusters, affinity={self.affinity_type}, fusion={self.fusion}")

        # Step 1: Sparse coefficients -> adjacency
        C = sparse_coefficients(
            X,
            gamma=self.gamma,
            affine=self.affine,
            solver=self.solver,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            K_dict=self.K_dict
        )

        Adj, memory_stats = coeffs_to_adjacency(
            C,
            K=self.K,
            use_sparse=self.use_sparse,
            eps=self._eps,
            zero_diag=self._zero_diag,
            normalize_cols=self._normalize_cols,
            verbose=self.verbose
        )
        if memory_stats:
            self.memory_stats_ = memory_stats

        # Step 2: Spectral affinity (RBF or self-tuning)
        if self.affinity_type == "self_tuning":
            A_sp = self_tuning_affinity(
                X,
                K_local=self.K_local,
                eps=self._eps,
                zero_diag=self._zero_diag,
                verbose=self.verbose
            )
        elif self.affinity_type == "rbf":
            A_sp = rbf_affinity(
                X,
                sigma=self.sigma,
                sparse_rbf=self.sparse_rbf,
                K_rbf=self.K_rbf,
                eps=self._eps,
                zero_diag=self._zero_diag,
                verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown affinity_type: {self.affinity_type}")

        # Step 3: Fusion
        A_fused = self._fuse_affinities(A_sp, Adj)

        # Step 4: Spectral embedding
        self.embedding_ = spectral_embedding(
            A_fused,
            n_clusters=self.n_clusters,
            eps=self._eps
        )

        # Step 5: KMeans on embedding
        km = KMeans(
            self.n_clusters,
            n_init=20,
            max_iter=1000,
            random_state=self.random_state
        )
        self.labels_ = km.fit_predict(self.embedding_)

        return self

    def fit_predict(self, X, y=None):
        """
        Fit model and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X, y)
        return self.labels_


# Alias for backward compatibility
S3FCParallel = S3FC
