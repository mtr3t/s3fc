"""
S3FC: Scalable Sparse Spectral Fusion Clustering
=================================================

A multi-manifold clustering method that fuses sparse subspace
affinity with spectral (RBF) affinity.

Example usage:
    from s3fc import S3FC

    model = S3FC(n_clusters=10, fusion='jordan', n_jobs=-1)
    labels = model.fit_predict(X)

Key features:
    - O(N*K) memory via sparse adjacency matrices (up to 188x savings)
    - Embarrassingly parallel coefficient solving via joblib
    - Adaptive fusion modes: 'jordan' for real data, 'legacy' for manifolds
    - No deep learning required - works directly on data

Module structure:
    - core.py: Main S3FC class (orchestration)
    - affinity.py: RBF kernel computation (dense/sparse)
    - coefficients.py: CVXPY sparse coefficient optimization
    - adjacency.py: Coefficient to adjacency conversion
    - embedding.py: Spectral embedding via eigendecomposition
"""

from .core import S3FC, S3FCParallel
from .affinity import rbf_affinity, dense_rbf_affinity, sparse_rbf_affinity, self_tuning_affinity
from .coefficients import sparse_coefficients, solve_single_coefficient
from .adjacency import coeffs_to_adjacency
from .embedding import spectral_embedding

# Shared utilities (Phase 1)
from .data_loader import load_dataset, load_legacy, list_available_datasets
from .metrics import nmi_score, ari_score, clustering_accuracy
from .toy_generator import (
    make_blobs,
    make_circles,
    make_moons,
    make_lines,
    make_crossing_lines,
    make_planes,
    make_spheres,
    make_mixed_geometry,
    make_varying_density,
    make_imbalanced,
    make_high_dimensional,
    # New generators matching legacy toy problems
    make_interlocking_circles,
    make_interlocking_curves,
    make_nested_circles,
    make_sinusoidal_plane,
    make_lines_and_planes,
    make_lines_and_sphere,
)
from .utils import validate_data, check_random_state, relabel_sequential, count_clusters

__version__ = "0.1.0"
__author__ = "Matthew T. Radice"
__all__ = [
    # Main class
    "S3FC",
    "S3FCParallel",
    # Affinity functions
    "rbf_affinity",
    "dense_rbf_affinity",
    "sparse_rbf_affinity",
    "self_tuning_affinity",
    # Coefficient functions
    "sparse_coefficients",
    "solve_single_coefficient",
    # Adjacency functions
    "coeffs_to_adjacency",
    # Embedding functions
    "spectral_embedding",
    # Data loading
    "load_dataset",
    "load_legacy",
    "list_available_datasets",
    # Metrics
    "nmi_score",
    "ari_score",
    "clustering_accuracy",
    # Toy generators
    "make_blobs",
    "make_circles",
    "make_moons",
    "make_lines",
    "make_crossing_lines",
    "make_planes",
    "make_spheres",
    "make_mixed_geometry",
    "make_varying_density",
    "make_imbalanced",
    "make_high_dimensional",
    "make_interlocking_circles",
    "make_interlocking_curves",
    "make_nested_circles",
    "make_sinusoidal_plane",
    "make_lines_and_planes",
    "make_lines_and_sphere",
    # Utilities
    "validate_data",
    "check_random_state",
    "relabel_sequential",
    "count_clusters",
]
