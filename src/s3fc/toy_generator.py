"""Synthetic cluster data generators for s3fc.

Provides configurable generators for testing clustering algorithms on
problems of varying difficulty: well-separated blobs, overlapping clusters,
manifolds (circles, moons), linear subspaces, and mixed geometries.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .utils import check_random_state, validate_data


def make_blobs(
    n_samples: int | list[int] = 100,
    n_features: int = 2,
    n_clusters: int = 3,
    cluster_std: float | list[float] = 1.0,
    center_box: tuple[float, float] = (-10.0, 10.0),
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate isotropic Gaussian blobs for clustering.

    Parameters
    ----------
    n_samples : int or list of int, default=100
        Total number of points, or list of points per cluster.
    n_features : int, default=2
        Number of features (dimensionality).
    n_clusters : int, default=3
        Number of clusters. Ignored if n_samples is a list.
    cluster_std : float or list of float, default=1.0
        Standard deviation of clusters. Higher = more spread/overlap.
    center_box : tuple of float, default=(-10.0, 10.0)
        Bounding box for cluster centers.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels.

    Examples
    --------
    >>> X, y = make_blobs(n_samples=300, n_clusters=3, random_state=42)
    >>> X.shape
    (300, 2)
    """
    rng = check_random_state(random_state)

    # Handle n_samples as list
    if isinstance(n_samples, list):
        n_samples_per_cluster = n_samples
        n_clusters = len(n_samples_per_cluster)
    else:
        n_samples_per_cluster = [n_samples // n_clusters] * n_clusters
        # Distribute remainder
        for i in range(n_samples % n_clusters):
            n_samples_per_cluster[i] += 1

    # Handle cluster_std as list
    if isinstance(cluster_std, (int, float)):
        cluster_std = [float(cluster_std)] * n_clusters

    # Generate cluster centers
    centers = rng.uniform(
        center_box[0], center_box[1], size=(n_clusters, n_features)
    )

    # Generate points
    X_list = []
    y_list = []
    for i, (n, std) in enumerate(zip(n_samples_per_cluster, cluster_std)):
        X_cluster = rng.randn(n, n_features) * std + centers[i]
        X_list.append(X_cluster)
        y_list.append(np.full(n, i, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_circles(
    n_samples: int = 100,
    noise: float = 0.0,
    factor: float = 0.8,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two concentric circles (2D).

    Parameters
    ----------
    n_samples : int, default=100
        Total number of points (split evenly between circles).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the data.
    factor : float, default=0.8
        Scale factor between inner and outer circle (0 < factor < 1).
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels (0 for outer, 1 for inner).
    """
    rng = check_random_state(random_state)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Outer circle
    theta_outer = rng.uniform(0, 2 * np.pi, n_outer)
    X_outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])

    # Inner circle
    theta_inner = rng.uniform(0, 2 * np.pi, n_inner)
    X_inner = np.column_stack([
        factor * np.cos(theta_inner),
        factor * np.sin(theta_inner),
    ])

    X = np.vstack([X_outer, X_inner])
    y = np.concatenate([np.zeros(n_outer), np.ones(n_inner)]).astype(np.int64)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_moons(
    n_samples: int = 100,
    noise: float = 0.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two interleaving half circles (moons) in 2D.

    Parameters
    ----------
    n_samples : int, default=100
        Total number of points (split evenly between moons).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the data.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels.
    """
    rng = check_random_state(random_state)

    n_upper = n_samples // 2
    n_lower = n_samples - n_upper

    # Upper moon
    theta_upper = np.linspace(0, np.pi, n_upper)
    X_upper = np.column_stack([np.cos(theta_upper), np.sin(theta_upper)])

    # Lower moon (shifted and flipped)
    theta_lower = np.linspace(0, np.pi, n_lower)
    X_lower = np.column_stack([
        1 - np.cos(theta_lower),
        0.5 - np.sin(theta_lower),
    ])

    X = np.vstack([X_upper, X_lower])
    y = np.concatenate([np.zeros(n_upper), np.ones(n_lower)]).astype(np.int64)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_two_lines_and_sphere(
    n_samples_per_structure: int = 100,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two crossing lines passing through a sphere (legacy toy problem #9).

    THE FLAGSHIP PROBLEM - two 1D linear subspaces piercing a 2D manifold (sphere).
    This challenging benchmark tests the ability to separate intersecting subspaces.

    Exact reproduction of the original S3FC toy problem:
    - Line 1 (red): from (0.2, 0.4, 0) to (0.8, 0.6, 1)
    - Line 2 (blue): from (0.2, 0.6, 1) to (0.8, 0.4, 0)
    - Sphere (green): center=(0.5, 0.5, 0.5), radius=0.2

    Parameters
    ----------
    n_samples_per_structure : int, default=100
        Number of points per structure (total = 3 * n_samples_per_structure).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to all coordinates.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Structure labels (0=line1, 1=line2, 2=sphere).
    """
    rng = check_random_state(random_state)
    n = n_samples_per_structure

    # Line parameters
    xp = np.linspace(0.2, 0.8, n)
    yp = np.linspace(0.4, 0.6, n)
    yn = np.linspace(0.6, 0.4, n)
    zp = np.linspace(0, 1, n)
    zn = np.linspace(1, 0, n)

    # Line 1: from (0.2, 0.4, 0) to (0.8, 0.6, 1)
    line1 = np.column_stack([xp, yp, zp])

    # Line 2: from (0.2, 0.6, 1) to (0.8, 0.4, 0)
    line2 = np.column_stack([xp, yn, zn])

    # Sphere: center=(0.5, 0.5, 0.5), radius=0.2
    # Original code uses hardcoded 10x10 grid, we use ceil(sqrt(n)) to handle any n
    center = [0.5, 0.5, 0.5]
    radius = 0.2
    # Use ceiling to ensure grid_size^2 >= n (avoids index errors for non-perfect squares)
    grid_size = int(np.ceil(np.sqrt(n)))  # 10 for n=100, 8 for n=50

    u, v = np.mgrid[0:2*np.pi:complex(0, grid_size), 0:np.pi:complex(0, grid_size)]
    x_sphere = radius * np.cos(u) * np.sin(v) + center[0]
    y_sphere = radius * np.sin(u) * np.sin(v) + center[1]
    z_sphere = radius * np.cos(v) + center[2]

    sphere = np.column_stack([
        x_sphere.flatten()[:n],
        y_sphere.flatten()[:n],
        z_sphere.flatten()[:n],
    ])

    # Add noise
    if noise > 0:
        line1 = line1 + rng.normal(0, noise, line1.shape)
        line2 = line2 + rng.normal(0, noise, line2.shape)
        sphere = sphere + rng.normal(0, noise, sphere.shape)

    X = np.vstack([line1, line2, sphere])
    y = np.concatenate([
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
        np.full(n, 2, dtype=np.int64),
    ])

    return validate_data(X, y)


def make_two_lines_and_two_sine_planes(
    n_samples_per_structure: int = 100,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two crossing lines and two sine planes in 3D (legacy toy problem #8b).

    Exact reproduction of the original S3FC toy problem:
    - Line 1 (red): from (0, 0.4, -1) to (1, 0.6, 1)
    - Line 2 (blue): from (0, 0.6, 1) to (1, 0.4, -1)
    - Sine Plane 1 (green): z = sin(x * 2π) * 0.5
    - Sine Plane 2 (magenta): z = sin(x * 2π) * -0.5 (inverted)

    Parameters
    ----------
    n_samples_per_structure : int, default=100
        Number of points per structure (total = 4 * n_samples_per_structure).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to all coordinates.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Structure labels (0=line1, 1=line2, 2=sine_plane1, 3=sine_plane2).
    """
    rng = check_random_state(random_state)
    n = n_samples_per_structure

    # Line parameters
    xp = np.linspace(0, 1, n)
    yp = np.linspace(0.4, 0.6, n)
    yn = np.linspace(0.6, 0.4, n)
    zp = np.linspace(-1, 1, n)
    zn = np.linspace(1, -1, n)

    # Line 1: from (0, 0.4, -1) to (1, 0.6, 1)
    line1 = np.column_stack([xp, yp, zp])

    # Line 2: from (0, 0.6, 1) to (1, 0.4, -1)
    line2 = np.column_stack([xp, yn, zn])

    # Sine plane grid
    grid_size = int(np.ceil(np.sqrt(n)))  # 10 for n=100, 8 for n=50
    px = np.linspace(0, 1, grid_size)
    px = np.tile(px, grid_size)
    py = np.linspace(0, 1, grid_size)
    py = np.repeat(py, grid_size)
    pzp = np.sin(px * np.pi * 2) * 0.5
    pzn = np.sin(px * np.pi * 2) * -0.5

    # Sine plane 1: z = sin(x * 2π) * 0.5
    sine_plane1 = np.column_stack([px, py, pzp])

    # Sine plane 2: z = sin(x * 2π) * -0.5 (inverted)
    sine_plane2 = np.column_stack([px, py, pzn])

    # Add noise
    if noise > 0:
        line1 = line1 + rng.normal(0, noise, line1.shape)
        line2 = line2 + rng.normal(0, noise, line2.shape)
        sine_plane1 = sine_plane1 + rng.normal(0, noise, sine_plane1.shape)
        sine_plane2 = sine_plane2 + rng.normal(0, noise, sine_plane2.shape)

    X = np.vstack([line1, line2, sine_plane1, sine_plane2])
    y = np.concatenate([
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
        np.full(n, 2, dtype=np.int64),
        np.full(n, 3, dtype=np.int64),
    ])

    return validate_data(X, y)


def make_two_lines_and_sine_plane(
    n_samples_per_structure: int = 100,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two crossing lines and one sine plane in 3D (legacy toy problem #8).

    Exact reproduction of the original S3FC toy problem:
    - Line 1 (red): from (0, 0.5, -1) to (1, 0.5, 1)
    - Line 2 (blue): from (0, 0.5, 1) to (1, 0.5, -1) - crosses Line 1
    - Sine Plane (green): 10x10 grid, z = sin(x * 2π) * 0.2

    Parameters
    ----------
    n_samples_per_structure : int, default=100
        Number of points per structure (total = 3 * n_samples_per_structure).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to all coordinates.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Structure labels (0=line1, 1=line2, 2=sine_plane).
    """
    rng = check_random_state(random_state)
    n = n_samples_per_structure

    # Line parameters
    x = np.linspace(0, 1, n)
    y_down = np.linspace(1, -1, n)
    ry_up = np.linspace(-1, 1, n)
    z_const = np.full(n, 0.5)

    # Line 1: (x, 0.5, ry) - from (0, 0.5, -1) to (1, 0.5, 1)
    line1 = np.column_stack([x, z_const, ry_up])

    # Line 2: (x, 0.5, y) - from (0, 0.5, 1) to (1, 0.5, -1)
    line2 = np.column_stack([x, z_const, y_down])

    # Sine plane: 10x10 grid, z = sin(x * 2π) * 0.2
    grid_size = int(np.ceil(np.sqrt(n)))  # 10 for n=100, 8 for n=50
    px = np.linspace(0, 1, grid_size)
    px = np.tile(px, grid_size)
    py = np.linspace(0, 1, grid_size)
    py = np.repeat(py, grid_size)
    pz = np.sin(px * np.pi * 2) * 0.2

    sine_plane = np.column_stack([px, py, pz])

    # Add noise
    if noise > 0:
        line1 = line1 + rng.normal(0, noise, line1.shape)
        line2 = line2 + rng.normal(0, noise, line2.shape)
        sine_plane = sine_plane + rng.normal(0, noise, sine_plane.shape)

    X = np.vstack([line1, line2, sine_plane])
    y = np.concatenate([
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
        np.full(n, 2, dtype=np.int64),
    ])

    return validate_data(X, y)


def make_two_lines_and_two_planes(
    n_samples_per_structure: int = 100,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two crossing lines and two tilted planes in 3D (legacy toy problem #7b).

    Exact reproduction of the original S3FC toy problem:
    - Line 1 (red): from (0, 0.4, 0) to (1, 0.6, 1)
    - Line 2 (blue): from (0, 0.6, 1) to (1, 0.4, 0)
    - Plane 1 (green): 10x10 grid, z tilted from 0.2 to 0.8
    - Plane 2 (magenta): 10x10 grid, z tilted from 0.8 to 0.2

    Parameters
    ----------
    n_samples_per_structure : int, default=100
        Number of points per structure (total = 4 * n_samples_per_structure).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to all coordinates.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Structure labels (0=line1, 1=line2, 2=plane1, 3=plane2).
    """
    rng = check_random_state(random_state)
    n = n_samples_per_structure

    # Line parameters
    xp = np.linspace(0, 1, n)
    xn = np.linspace(1, 0, n)
    yp = np.linspace(0.4, 0.6, n)
    yn = np.linspace(0.6, 0.4, n)
    zp = np.linspace(0, 1, n)
    zn = np.linspace(1, 0, n)

    # Line 1: (xp, yp, zp) - from (0, 0.4, 0) to (1, 0.6, 1)
    line1 = np.column_stack([xp, yp, zp])

    # Line 2: (xp, yn, zn) - from (0, 0.6, 1) to (1, 0.4, 0)
    line2 = np.column_stack([xp, yn, zn])

    # Plane grid
    grid_size = int(np.ceil(np.sqrt(n)))  # 10 for n=100, 8 for n=50
    px = np.linspace(0, 1, grid_size)
    px = np.tile(px, grid_size)
    py = np.linspace(0, 1, grid_size)
    py = np.repeat(py, grid_size)
    pzp = np.linspace(0.2, 0.8, n)
    pzn = np.linspace(0.8, 0.2, n)

    # Plane 1: tilted up
    plane1 = np.column_stack([px, py, pzp])

    # Plane 2: tilted down
    plane2 = np.column_stack([px, py, pzn])

    # Add noise
    if noise > 0:
        line1 = line1 + rng.normal(0, noise, line1.shape)
        line2 = line2 + rng.normal(0, noise, line2.shape)
        plane1 = plane1 + rng.normal(0, noise, plane1.shape)
        plane2 = plane2 + rng.normal(0, noise, plane2.shape)

    X = np.vstack([line1, line2, plane1, plane2])
    y = np.concatenate([
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
        np.full(n, 2, dtype=np.int64),
        np.full(n, 3, dtype=np.int64),
    ])

    return validate_data(X, y)


def make_two_lines_and_plane(
    n_samples_per_structure: int = 100,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two crossing lines and a plane in 3D (legacy toy problem #7).

    Exact reproduction of the original S3FC toy problem:
    - Line 1 (red): (t, 0.5, t) for t in [0,1] - diagonal X=Z at Y=0.5
    - Line 2 (blue): (t, 0.5, 1-t) for t in [0,1] - diagonal Z=1-X at Y=0.5
    - Plane (green): Z=0.5, 10x10 grid in XY plane

    Parameters
    ----------
    n_samples_per_structure : int, default=100
        Number of points per structure (total = 3 * n_samples_per_structure).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to all coordinates.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Structure labels (0=line1, 1=line2, 2=plane).
    """
    rng = check_random_state(random_state)
    n = n_samples_per_structure

    # Line parameters
    x = np.linspace(0, 1, n)
    y_line = np.linspace(1, 0, n)
    z = np.full(n, 0.5)

    # Line 1: (x, 0.5, x) - diagonal where X=Z
    line1 = np.column_stack([x, z, x])

    # Line 2: (x, 0.5, y) where y goes 1->0 - diagonal where Z=1-X
    line2 = np.column_stack([x, z, y_line])

    # Plane: Z=0.5, 10x10 grid in XY
    grid_size = int(np.ceil(np.sqrt(n)))  # 10 for n=100, 8 for n=50
    px = np.linspace(0, 1, grid_size)
    px = np.tile(px, grid_size)
    py = np.linspace(0, 1, grid_size)
    py = np.repeat(py, grid_size)
    pz = np.full(n, 0.5)
    plane = np.column_stack([px, py, pz])

    # Add noise
    if noise > 0:
        line1 = line1 + rng.normal(0, noise, line1.shape)
        line2 = line2 + rng.normal(0, noise, line2.shape)
        plane = plane + rng.normal(0, noise, plane.shape)

    X = np.vstack([line1, line2, plane])
    y = np.concatenate([
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
        np.full(n, 2, dtype=np.int64),
    ])

    return validate_data(X, y)


def make_two_crossing_lines_x(
    n_samples_per_line: int = 100,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two crossing lines forming an X (legacy toy problem #6).

    Exact reproduction of the original S3FC toy problem:
    - Line 1 (red): y = x, from (-50, -50) to (49, 49)
    - Line 2 (blue): y = -x, from (-50, 50) to (49, -49)
    - Lines cross at origin (0, 0)

    Parameters
    ----------
    n_samples_per_line : int, default=100
        Number of points per line (total = 2 * n_samples_per_line).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to both coordinates.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Line labels (0 or 1).
    """
    rng = check_random_state(random_state)
    n = n_samples_per_line

    # Line 1: y = x, from -50 to 49
    t = np.arange(-50, -50 + n)  # -50, -49, ..., 49
    x1 = t.astype(float)
    y1 = t.astype(float)

    # Line 2: y = -x, from -50 to 49
    x2 = t.astype(float)
    y2 = -t.astype(float)

    # Add noise to both coordinates
    if noise > 0:
        x1 = x1 + rng.normal(0, noise, n)
        y1 = y1 + rng.normal(0, noise, n)
        x2 = x2 + rng.normal(0, noise, n)
        y2 = y2 + rng.normal(0, noise, n)

    X1 = np.column_stack([x1, y1])
    X2 = np.column_stack([x2, y2])

    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    return validate_data(X, y)


def make_two_subset_circles(
    n_samples_per_circle: int = 100,
    inner_radius: float = 2.0,
    outer_radius: float = 6.0,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two concentric circles - one inside another (legacy toy problem #5).

    Exact reproduction of the original S3FC toy problem:
    - Circle 1 (inner/red): radius=2, center=(0, 0)
    - Circle 2 (outer/blue): radius=6, center=(0, 0)
    - Points evenly spaced around each circle

    Parameters
    ----------
    n_samples_per_circle : int, default=100
        Number of points per circle (total = 2 * n_samples_per_circle).
    inner_radius : float, default=2.0
        Radius of inner circle.
    outer_radius : float, default=6.0
        Radius of outer circle.
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to both coordinates.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Circle labels (0=inner, 1=outer).
    """
    rng = check_random_state(random_state)
    n = n_samples_per_circle

    # Angles: 2*pi/n * i for i in 0..n-1
    angles = np.array([2 * np.pi / n * i for i in range(n)])

    # Inner circle (radius=2)
    x1 = np.cos(angles) * inner_radius
    y1 = np.sin(angles) * inner_radius

    # Outer circle (radius=6)
    x2 = np.cos(angles) * outer_radius
    y2 = np.sin(angles) * outer_radius

    # Add noise
    if noise > 0:
        x1 = x1 + rng.normal(0, noise, n)
        y1 = y1 + rng.normal(0, noise, n)
        x2 = x2 + rng.normal(0, noise, n)
        y2 = y2 + rng.normal(0, noise, n)

    X1 = np.column_stack([x1, y1])
    X2 = np.column_stack([x2, y2])

    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    return validate_data(X, y)


def make_two_interlocking_curves(
    n_samples_per_curve: int = 100,
    radius: float = 2.0,
    vertical_shift: float = 3.0,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two interlocking curves/semicircles (legacy toy problem #4).

    Exact reproduction of the original S3FC toy problem:
    - Curve 1 (red): Bottom semicircle of circle at (0, 3)
    - Curve 2 (blue): Top semicircle of circle at (0, 0)
    - Both have radius=2
    - Points from 200-point circle, using indices 100-199 and 0-99

    Parameters
    ----------
    n_samples_per_curve : int, default=100
        Number of points per curve (total = 2 * n_samples_per_curve).
    radius : float, default=2.0
        Radius of both semicircles.
    vertical_shift : float, default=3.0
        Vertical shift of curve 1's center.
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to both coordinates.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Curve labels (0 or 1).
    """
    rng = check_random_state(random_state)
    n = n_samples_per_curve

    # Generate full circles with 2*n points (matching original)
    n_full = 2 * n
    angles_full = np.array([2 * np.pi / n_full * i for i in range(n_full + 1)])

    # Curve 1: bottom semicircle shifted up (indices n to 2n-1)
    # Circle centered at (0, vertical_shift)
    angles1 = angles_full[n:2*n]  # indices 100-199 for n=100
    x1 = np.cos(angles1) * radius
    y1 = np.sin(angles1) * radius + vertical_shift

    # Curve 2: top semicircle at origin (indices 0 to n-1)
    angles2 = angles_full[0:n]  # indices 0-99 for n=100
    x2 = np.cos(angles2) * radius
    y2 = np.sin(angles2) * radius

    # Add noise
    if noise > 0:
        x1 = x1 + rng.normal(0, noise, n)
        y1 = y1 + rng.normal(0, noise, n)
        x2 = x2 + rng.normal(0, noise, n)
        y2 = y2 + rng.normal(0, noise, n)

    X1 = np.column_stack([x1, y1])
    X2 = np.column_stack([x2, y2])

    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    return validate_data(X, y)


def make_two_interlocking_circles_legacy(
    n_samples_per_circle: int = 100,
    radius: float = 2.0,
    shift: float = 2.0,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two interlocking circles (legacy toy problem #2).

    Exact reproduction of the original S3FC toy problem:
    - Circle 1: center=(0, 0), radius=2
    - Circle 2: center=(2, 0), radius=2 (shifted right)
    - Points evenly spaced around each circle
    - Noise added to both x and y coordinates

    Parameters
    ----------
    n_samples_per_circle : int, default=100
        Number of points per circle (total = 2 * n_samples_per_circle).
    radius : float, default=2.0
        Radius of both circles.
    shift : float, default=2.0
        Horizontal shift of circle 2 (center at x=shift).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to both coordinates.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Circle labels (0 or 1).
    """
    rng = check_random_state(random_state)
    n = n_samples_per_circle

    # Circle 1: center at origin
    angles1 = np.array([2 * np.pi / n * i for i in range(n)])
    x1 = np.cos(angles1) * radius
    y1 = np.sin(angles1) * radius

    # Circle 2: center shifted right
    angles2 = np.array([2 * np.pi / n * i for i in range(n)])
    x2 = np.cos(angles2) * radius + shift
    y2 = np.sin(angles2) * radius

    # Add noise to both coordinates
    if noise > 0:
        x1 = x1 + rng.normal(0, noise, n)
        y1 = y1 + rng.normal(0, noise, n)
        x2 = x2 + rng.normal(0, noise, n)
        y2 = y2 + rng.normal(0, noise, n)

    X1 = np.column_stack([x1, y1])
    X2 = np.column_stack([x2, y2])

    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    return validate_data(X, y)


def make_two_horizontal_lines(
    n_samples_per_line: int = 100,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two horizontal lines (legacy toy problem #1).

    Exact reproduction of the original S3FC toy problem:
    - Line 1: x from 0.1 to 10.0, y = 0
    - Line 2: x from 0.1 to 10.0, y = 5
    - Noise added to x-coordinate only
    - Points are evenly spaced (linspace), not random

    Parameters
    ----------
    n_samples_per_line : int, default=100
        Number of points per line (total = 2 * n_samples_per_line).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to x-coordinate.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Line labels (0 or 1).

    Examples
    --------
    >>> X, y = make_two_horizontal_lines(n_samples_per_line=100, noise=0.0)
    >>> X.shape
    (200, 2)
    """
    rng = check_random_state(random_state)

    n = n_samples_per_line

    # Line 1: x from 0.1 to 10.0, y = 0
    x1 = np.linspace(0.1, 10.0, n)
    if noise > 0:
        x1 = x1 + rng.normal(0, noise, n)
    y1_coord = np.zeros(n)
    X1 = np.column_stack([x1, y1_coord])

    # Line 2: x from 0.1 to 10.0, y = 5
    x2 = np.linspace(0.1, 10.0, n)
    if noise > 0:
        x2 = x2 + rng.normal(0, noise, n)
    y2_coord = np.full(n, 5.0)
    X2 = np.column_stack([x2, y2_coord])

    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    return validate_data(X, y)


def make_lines(
    n_samples: int = 100,
    n_lines: int = 2,
    n_features: int = 2,
    noise: float = 0.0,
    separation: float = 1.0,
    line_length: float = 10.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate parallel lines (linear subspaces).

    Parameters
    ----------
    n_samples : int, default=100
        Total number of points.
    n_lines : int, default=2
        Number of parallel lines.
    n_features : int, default=2
        Number of features (dimensionality).
    noise : float, default=0.0
        Standard deviation of Gaussian noise perpendicular to lines.
    separation : float, default=1.0
        Distance between adjacent lines.
    line_length : float, default=10.0
        Length of lines along the principal axis.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Line labels.
    """
    rng = check_random_state(random_state)

    n_per_line = [n_samples // n_lines] * n_lines
    for i in range(n_samples % n_lines):
        n_per_line[i] += 1

    X_list = []
    y_list = []

    for i, n in enumerate(n_per_line):
        # Points along the line (first dimension)
        t = rng.uniform(0, line_length, n)
        X_line = np.zeros((n, n_features))
        X_line[:, 0] = t

        # Offset in second dimension for separation
        if n_features >= 2:
            X_line[:, 1] = i * separation

        # Add noise in all dimensions
        if noise > 0:
            X_line += rng.randn(n, n_features) * noise

        X_list.append(X_line)
        y_list.append(np.full(n, i, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_crossing_lines(
    n_samples: int = 100,
    n_features: int = 2,
    noise: float = 0.0,
    angle: float = 90.0,
    line_length: float = 10.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two crossing lines at a specified angle.

    Parameters
    ----------
    n_samples : int, default=100
        Total number of points.
    n_features : int, default=2
        Number of features (minimum 2).
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    angle : float, default=90.0
        Angle between lines in degrees.
    line_length : float, default=10.0
        Length of each line.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Line labels.
    """
    rng = check_random_state(random_state)

    n_line1 = n_samples // 2
    n_line2 = n_samples - n_line1

    # Line 1: along x-axis
    t1 = rng.uniform(-line_length / 2, line_length / 2, n_line1)
    X1 = np.zeros((n_line1, n_features))
    X1[:, 0] = t1

    # Line 2: rotated by angle
    theta = np.deg2rad(angle)
    t2 = rng.uniform(-line_length / 2, line_length / 2, n_line2)
    X2 = np.zeros((n_line2, n_features))
    X2[:, 0] = t2 * np.cos(theta)
    X2[:, 1] = t2 * np.sin(theta)

    X = np.vstack([X1, X2])
    y = np.concatenate([
        np.zeros(n_line1, dtype=np.int64),
        np.ones(n_line2, dtype=np.int64),
    ])

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_planes(
    n_samples: int = 100,
    n_planes: int = 2,
    n_features: int = 3,
    noise: float = 0.0,
    separation: float = 2.0,
    plane_size: float = 10.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate parallel planes (2D linear subspaces in higher dimensions).

    Parameters
    ----------
    n_samples : int, default=100
        Total number of points.
    n_planes : int, default=2
        Number of parallel planes.
    n_features : int, default=3
        Number of features (minimum 3).
    noise : float, default=0.0
        Standard deviation of Gaussian noise perpendicular to planes.
    separation : float, default=2.0
        Distance between adjacent planes.
    plane_size : float, default=10.0
        Size of planes in each in-plane dimension.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Plane labels.
    """
    if n_features < 3:
        raise ValueError(f"make_planes requires n_features >= 3, got {n_features}")

    rng = check_random_state(random_state)

    n_per_plane = [n_samples // n_planes] * n_planes
    for i in range(n_samples % n_planes):
        n_per_plane[i] += 1

    X_list = []
    y_list = []

    for i, n in enumerate(n_per_plane):
        # Points in the plane (first two dimensions)
        X_plane = np.zeros((n, n_features))
        X_plane[:, 0] = rng.uniform(0, plane_size, n)
        X_plane[:, 1] = rng.uniform(0, plane_size, n)

        # Offset in third dimension for separation
        X_plane[:, 2] = i * separation

        # Add noise
        if noise > 0:
            X_plane += rng.randn(n, n_features) * noise

        X_list.append(X_plane)
        y_list.append(np.full(n, i, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_spheres(
    n_samples: int = 100,
    n_features: int = 3,
    noise: float = 0.0,
    factor: float = 0.5,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two concentric spheres (3D analog of circles).

    Parameters
    ----------
    n_samples : int, default=100
        Total number of points (split evenly between spheres).
    n_features : int, default=3
        Number of features (minimum 3).
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    factor : float, default=0.5
        Scale factor between inner and outer sphere (0 < factor < 1).
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels (0 for outer, 1 for inner).
    """
    if n_features < 3:
        raise ValueError(f"make_spheres requires n_features >= 3, got {n_features}")

    rng = check_random_state(random_state)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Generate points on unit sphere using normal distribution
    def random_sphere(n, radius):
        points = rng.randn(n, n_features)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points * radius

    X_outer = random_sphere(n_outer, 1.0)
    X_inner = random_sphere(n_inner, factor)

    X = np.vstack([X_outer, X_inner])
    y = np.concatenate([np.zeros(n_outer), np.ones(n_inner)]).astype(np.int64)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_mixed_geometry(
    n_samples: int = 300,
    n_features: int = 3,
    components: list[Literal["line", "plane", "sphere"]] | None = None,
    noise: float = 0.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate data with mixed geometric structures.

    Combines lines, planes, and spheres into one dataset. Useful for
    testing multi-manifold clustering algorithms.

    Parameters
    ----------
    n_samples : int, default=300
        Total number of points (split among components).
    n_features : int, default=3
        Number of features (minimum 3).
    components : list of str, optional
        List of component types: 'line', 'plane', 'sphere'.
        Default is ['line', 'line', 'plane'] (two lines and a plane).
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Component labels.
    """
    if n_features < 3:
        raise ValueError(f"make_mixed_geometry requires n_features >= 3, got {n_features}")

    if components is None:
        components = ["line", "line", "plane"]

    rng = check_random_state(random_state)

    n_components = len(components)
    n_per_component = [n_samples // n_components] * n_components
    for i in range(n_samples % n_components):
        n_per_component[i] += 1

    X_list = []
    y_list = []

    for i, (comp_type, n) in enumerate(zip(components, n_per_component)):
        X_comp = np.zeros((n, n_features))

        if comp_type == "line":
            # Line along a random direction
            direction = rng.randn(n_features)
            direction /= np.linalg.norm(direction)
            t = rng.uniform(-5, 5, n)
            X_comp = np.outer(t, direction)
            # Shift to avoid overlap
            offset = rng.randn(n_features) * 3
            X_comp += offset

        elif comp_type == "plane":
            # Plane spanned by two random orthogonal vectors
            v1 = rng.randn(n_features)
            v1 /= np.linalg.norm(v1)
            v2 = rng.randn(n_features)
            v2 -= np.dot(v2, v1) * v1  # Orthogonalize
            v2 /= np.linalg.norm(v2)

            t1 = rng.uniform(-5, 5, n)
            t2 = rng.uniform(-5, 5, n)
            X_comp = np.outer(t1, v1) + np.outer(t2, v2)
            # Shift
            offset = rng.randn(n_features) * 3
            X_comp += offset

        elif comp_type == "sphere":
            # Random points on a sphere
            X_comp = rng.randn(n, n_features)
            X_comp /= np.linalg.norm(X_comp, axis=1, keepdims=True)
            X_comp *= 2  # Radius 2
            # Shift
            offset = rng.randn(n_features) * 5
            X_comp += offset

        else:
            raise ValueError(f"Unknown component type: {comp_type}")

        X_list.append(X_comp)
        y_list.append(np.full(n, i, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_varying_density(
    n_samples: int = 300,
    n_clusters: int = 3,
    n_features: int = 2,
    density_range: tuple[float, float] = (0.5, 2.0),
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate clusters with varying densities.

    Some clusters are tight, others are diffuse. Challenging for algorithms
    that assume uniform density.

    Parameters
    ----------
    n_samples : int, default=300
        Total number of points.
    n_clusters : int, default=3
        Number of clusters.
    n_features : int, default=2
        Number of features.
    density_range : tuple of float, default=(0.5, 2.0)
        Range of cluster standard deviations.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels.
    """
    rng = check_random_state(random_state)

    # Varying standard deviations
    stds = rng.uniform(density_range[0], density_range[1], n_clusters)

    return make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        cluster_std=list(stds),
        shuffle=shuffle,
        random_state=rng,
    )


def make_imbalanced(
    n_samples: int = 300,
    n_clusters: int = 3,
    n_features: int = 2,
    imbalance_ratio: float = 10.0,
    cluster_std: float = 1.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate clusters with imbalanced sizes.

    The largest cluster has imbalance_ratio times more points than the smallest.

    Parameters
    ----------
    n_samples : int, default=300
        Total number of points.
    n_clusters : int, default=3
        Number of clusters.
    n_features : int, default=2
        Number of features.
    imbalance_ratio : float, default=10.0
        Ratio of largest to smallest cluster size.
    cluster_std : float, default=1.0
        Standard deviation of clusters.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels.
    """
    rng = check_random_state(random_state)

    # Generate cluster sizes following a geometric progression
    ratios = np.geomspace(1, imbalance_ratio, n_clusters)
    ratios = ratios / ratios.sum()
    sizes = (ratios * n_samples).astype(int)
    # Adjust to match exact n_samples
    sizes[-1] = n_samples - sizes[:-1].sum()

    return make_blobs(
        n_samples=list(sizes),
        n_features=n_features,
        cluster_std=cluster_std,
        shuffle=shuffle,
        random_state=rng,
    )


def make_high_dimensional(
    n_samples: int = 200,
    n_clusters: int = 3,
    n_features: int = 100,
    n_informative: int = 10,
    cluster_std: float = 1.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate high-dimensional data with clusters in a low-dimensional subspace.

    Only n_informative features contain cluster structure; the rest are noise.

    Parameters
    ----------
    n_samples : int, default=200
        Total number of points.
    n_clusters : int, default=3
        Number of clusters.
    n_features : int, default=100
        Total number of features.
    n_informative : int, default=10
        Number of features that contain cluster information.
    cluster_std : float, default=1.0
        Standard deviation of clusters in informative dimensions.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels.
    """
    rng = check_random_state(random_state)

    # Generate clusters in informative dimensions
    X_info, y = make_blobs(
        n_samples=n_samples,
        n_features=n_informative,
        n_clusters=n_clusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=rng,
    )

    # Add noise dimensions
    n_noise = n_features - n_informative
    X_noise = rng.randn(n_samples, n_noise)

    X = np.hstack([X_info, X_noise])

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_interlocking_circles(
    n_samples: int = 100,
    noise: float = 0.0,
    separation: float = 1.0,
    radius: float = 2.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two overlapping/interlocking circles (Venn diagram style).

    Two circles that partially overlap, creating a challenging separation problem.

    Parameters
    ----------
    n_samples : int, default=100
        Total number of points (split evenly between circles).
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    separation : float, default=1.0
        Distance between circle centers. Smaller = more overlap.
    radius : float, default=2.0
        Radius of both circles.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels.
    """
    rng = check_random_state(random_state)

    n_per_circle = n_samples // 2
    n_circle2 = n_samples - n_per_circle

    # Circle 1: centered at origin
    theta1 = np.linspace(0, 2 * np.pi, n_per_circle, endpoint=False)
    X1 = np.column_stack([
        radius * np.cos(theta1),
        radius * np.sin(theta1),
    ])

    # Circle 2: offset by separation
    theta2 = np.linspace(0, 2 * np.pi, n_circle2, endpoint=False)
    X2 = np.column_stack([
        radius * np.cos(theta2) + separation,
        radius * np.sin(theta2),
    ])

    X = np.vstack([X1, X2])
    y = np.concatenate([
        np.zeros(n_per_circle, dtype=np.int64),
        np.ones(n_circle2, dtype=np.int64),
    ])

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_interlocking_curves(
    n_samples: int = 200,
    radius: float = 2.0,
    vertical_shift: float = 3.0,
    noise: float = 0.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate two interlocking semicircles.

    Creates two semicircular curves that interlock/overlap:
    - Curve 1: Bottom half of a circle shifted up
    - Curve 2: Top half of a circle at origin

    This matches the legacy "03_two_interlocking_curves" toy problem.

    Parameters
    ----------
    n_samples : int, default=200
        Total number of points (split evenly between curves).
    radius : float, default=2.0
        Radius of both semicircles.
    vertical_shift : float, default=3.0
        Vertical shift for the upper semicircle.
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels.
    """
    rng = check_random_state(random_state)

    n_per_curve = n_samples // 2
    n_curve2 = n_samples - n_per_curve

    # Curve 1: Bottom semicircle of a circle shifted up by vertical_shift
    # (angles from pi to 2*pi give the bottom half)
    theta1 = np.linspace(np.pi, 2 * np.pi, n_per_curve)
    X1 = np.column_stack([
        radius * np.cos(theta1),
        radius * np.sin(theta1) + vertical_shift,
    ])

    # Curve 2: Top semicircle at origin
    # (angles from 0 to pi give the top half)
    theta2 = np.linspace(0, np.pi, n_curve2)
    X2 = np.column_stack([
        radius * np.cos(theta2),
        radius * np.sin(theta2),
    ])

    X = np.vstack([X1, X2])
    y = np.concatenate([
        np.zeros(n_per_curve, dtype=np.int64),
        np.ones(n_curve2, dtype=np.int64),
    ])

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_nested_circles(
    n_samples: int = 200,
    noise: float = 0.0,
    inner_radius: float = 2.0,
    outer_radius: float = 6.0,
    offset: tuple[float, float] = (0.0, 0.0),
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate nested circles (one inside another, possibly offset).

    Also known as "subset circles" - a small circle inside a larger one.
    This matches the legacy "04_two_subset_circles" toy problem.

    Parameters
    ----------
    n_samples : int, default=100
        Total number of points (split evenly between circles).
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    inner_radius : float, default=0.5
        Radius of inner circle.
    outer_radius : float, default=2.0
        Radius of outer circle.
    offset : tuple of float, default=(0.0, 0.0)
        (x, y) offset of inner circle center from outer circle center.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels (0 for outer, 1 for inner).
    """
    rng = check_random_state(random_state)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Outer circle
    theta_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    X_outer = np.column_stack([
        outer_radius * np.cos(theta_outer),
        outer_radius * np.sin(theta_outer),
    ])

    # Inner circle (with offset)
    theta_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    X_inner = np.column_stack([
        inner_radius * np.cos(theta_inner) + offset[0],
        inner_radius * np.sin(theta_inner) + offset[1],
    ])

    X = np.vstack([X_outer, X_inner])
    y = np.concatenate([
        np.zeros(n_outer, dtype=np.int64),
        np.ones(n_inner, dtype=np.int64),
    ])

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_sinusoidal_plane(
    n_samples: int = 100,
    n_features: int = 3,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    plane_size: float = 10.0,
    noise: float = 0.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate a sinusoidal (wavy) plane surface.

    A 2D surface in 3D space with sinusoidal variation in the z-dimension.

    Parameters
    ----------
    n_samples : int, default=100
        Number of points on the surface.
    n_features : int, default=3
        Number of features (minimum 3).
    amplitude : float, default=1.0
        Amplitude of the sinusoidal wave.
    frequency : float, default=1.0
        Frequency of the sinusoidal wave.
    plane_size : float, default=10.0
        Size of the plane in x and y dimensions.
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Labels (all zeros - single cluster).
    """
    if n_features < 3:
        raise ValueError(f"make_sinusoidal_plane requires n_features >= 3, got {n_features}")

    rng = check_random_state(random_state)

    # Generate grid points
    X = np.zeros((n_samples, n_features))
    X[:, 0] = rng.uniform(0, plane_size, n_samples)
    X[:, 1] = rng.uniform(0, plane_size, n_samples)
    # Sinusoidal z-coordinate
    X[:, 2] = amplitude * np.sin(frequency * X[:, 0]) * np.sin(frequency * X[:, 1])

    y = np.zeros(n_samples, dtype=np.int64)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_lines_and_planes(
    n_samples: int = 300,
    n_lines: int = 2,
    n_planes: int = 1,
    n_features: int = 3,
    sinusoidal: bool = False,
    noise: float = 0.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate a mix of lines and planes in 3D (matching legacy toy problems).

    Creates the classic multi-manifold clustering test case with
    1D subspaces (lines) and 2D subspaces (planes).

    Parameters
    ----------
    n_samples : int, default=300
        Total number of points.
    n_lines : int, default=2
        Number of lines.
    n_planes : int, default=1
        Number of planes.
    n_features : int, default=3
        Number of features (minimum 3).
    sinusoidal : bool, default=False
        If True, planes are sinusoidal (wavy) instead of flat.
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels.
    """
    if n_features < 3:
        raise ValueError(f"make_lines_and_planes requires n_features >= 3, got {n_features}")

    rng = check_random_state(random_state)

    n_components = n_lines + n_planes
    n_per_component = [n_samples // n_components] * n_components
    for i in range(n_samples % n_components):
        n_per_component[i] += 1

    X_list = []
    y_list = []

    # Generate lines
    for i in range(n_lines):
        n = n_per_component[i]
        # Line along a specific direction
        direction = np.zeros(n_features)
        direction[i % n_features] = 1.0
        direction[(i + 1) % n_features] = 0.5
        direction /= np.linalg.norm(direction)

        t = np.linspace(-5, 5, n)
        X_line = np.outer(t, direction)

        # Offset each line
        offset = np.zeros(n_features)
        offset[(i + 2) % n_features] = i * 3
        X_line += offset

        X_list.append(X_line)
        y_list.append(np.full(n, i, dtype=np.int64))

    # Generate planes
    for i in range(n_planes):
        n = n_per_component[n_lines + i]
        X_plane = np.zeros((n, n_features))

        # Grid in first two dimensions
        X_plane[:, 0] = rng.uniform(-5, 5, n)
        X_plane[:, 1] = rng.uniform(-5, 5, n)

        if sinusoidal:
            # Wavy plane
            X_plane[:, 2] = np.sin(X_plane[:, 0]) * np.cos(X_plane[:, 1])
        else:
            # Flat plane at z=0
            X_plane[:, 2] = 0

        # Offset each plane
        offset = np.zeros(n_features)
        offset[2] = (i + 1) * 4
        X_plane += offset

        X_list.append(X_plane)
        y_list.append(np.full(n, n_lines + i, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_swiss_roll(
    n_samples: int = 1000,
    noise: float = 0.0,
    n_turns: float = 1.5,
    hole: bool = False,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate the classic Swiss Roll manifold in 3D.

    The Swiss Roll is a 2D manifold embedded in 3D space, commonly used
    for testing manifold learning and clustering algorithms.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of points on the manifold.
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the data.
    n_turns : float, default=1.5
        Number of turns of the spiral.
    hole : bool, default=False
        If True, creates a hole in the center of the roll.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Labels (continuous values along the manifold, discretized to 3 clusters).

    Notes
    -----
    This is a single-manifold dataset. S3FC is designed for multi-manifold
    clustering and may not be optimal for this case. Included for completeness
    and comparison with RMMC paper results.

    Examples
    --------
    >>> X, y = make_swiss_roll(n_samples=500, random_state=42)
    >>> X.shape
    (500, 3)
    >>> len(np.unique(y))
    3
    """
    rng = check_random_state(random_state)

    # Generate t values (position along the roll)
    if hole:
        t = rng.uniform(3 * np.pi / 2, 4 * np.pi * n_turns, n_samples)
    else:
        t = rng.uniform(1.5 * np.pi, 1.5 * np.pi + 2 * np.pi * n_turns, n_samples)

    # Height along the roll
    height = rng.uniform(0, 21, n_samples)

    # Swiss roll parametric equations
    X = np.zeros((n_samples, 3))
    X[:, 0] = t * np.cos(t)
    X[:, 1] = height
    X[:, 2] = t * np.sin(t)

    # Add noise
    if noise > 0:
        X += rng.randn(*X.shape) * noise

    # Discretize t into 3 clusters for labels
    t_min, t_max = t.min(), t.max()
    t_normalized = (t - t_min) / (t_max - t_min)
    y = (t_normalized * 3).astype(np.int64)
    y = np.clip(y, 0, 2)  # Ensure labels are 0, 1, 2

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_three_planes(
    n_samples: int = 300,
    n_features: int = 3,
    noise: float = 0.0,
    plane_size: float = 5.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate three intersecting 2D planes in 3D space.

    Creates three planes that intersect at the origin, a classic
    multi-manifold clustering benchmark used in the RMMC paper.

    The planes are:
    - Plane 0: XY plane (z=0)
    - Plane 1: XZ plane (y=0)
    - Plane 2: YZ plane (x=0)

    Parameters
    ----------
    n_samples : int, default=300
        Total number of points (split evenly among planes).
    n_features : int, default=3
        Number of features (minimum 3).
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the data.
    plane_size : float, default=5.0
        Size of each plane (points are generated in [-plane_size, plane_size]).
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Plane labels (0, 1, or 2).

    Notes
    -----
    RMMC paper reports ACC=0.733 on this benchmark. This is a challenging
    case due to the planes intersecting at the origin.

    Examples
    --------
    >>> X, y = make_three_planes(n_samples=300, random_state=42)
    >>> X.shape
    (300, 3)
    >>> len(np.unique(y))
    3
    """
    if n_features < 3:
        raise ValueError(f"make_three_planes requires n_features >= 3, got {n_features}")

    rng = check_random_state(random_state)

    n_per_plane = [n_samples // 3] * 3
    for i in range(n_samples % 3):
        n_per_plane[i] += 1

    X_list = []
    y_list = []

    # Plane 0: XY plane (z=0)
    n = n_per_plane[0]
    X_xy = np.zeros((n, n_features))
    X_xy[:, 0] = rng.uniform(-plane_size, plane_size, n)
    X_xy[:, 1] = rng.uniform(-plane_size, plane_size, n)
    X_list.append(X_xy)
    y_list.append(np.full(n, 0, dtype=np.int64))

    # Plane 1: XZ plane (y=0)
    n = n_per_plane[1]
    X_xz = np.zeros((n, n_features))
    X_xz[:, 0] = rng.uniform(-plane_size, plane_size, n)
    X_xz[:, 2] = rng.uniform(-plane_size, plane_size, n)
    X_list.append(X_xz)
    y_list.append(np.full(n, 1, dtype=np.int64))

    # Plane 2: YZ plane (x=0)
    n = n_per_plane[2]
    X_yz = np.zeros((n, n_features))
    X_yz[:, 1] = rng.uniform(-plane_size, plane_size, n)
    X_yz[:, 2] = rng.uniform(-plane_size, plane_size, n)
    X_list.append(X_yz)
    y_list.append(np.full(n, 2, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_lines_and_sphere(
    n_samples: int = 300,
    n_lines: int = 2,
    n_features: int = 3,
    sphere_radius: float = 2.0,
    noise: float = 0.0,
    shuffle: bool = True,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate lines and a sphere in 3D (matching legacy toy problem 10).

    Creates the classic multi-manifold clustering test case with
    1D subspaces (lines) and a 2D manifold (sphere surface).

    Parameters
    ----------
    n_samples : int, default=300
        Total number of points.
    n_lines : int, default=2
        Number of lines.
    n_features : int, default=3
        Number of features (minimum 3).
    sphere_radius : float, default=2.0
        Radius of the sphere.
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Cluster labels.
    """
    if n_features < 3:
        raise ValueError(f"make_lines_and_sphere requires n_features >= 3, got {n_features}")

    rng = check_random_state(random_state)

    n_components = n_lines + 1  # lines + sphere
    n_per_component = [n_samples // n_components] * n_components
    for i in range(n_samples % n_components):
        n_per_component[i] += 1

    X_list = []
    y_list = []

    # Generate lines
    for i in range(n_lines):
        n = n_per_component[i]
        direction = np.zeros(n_features)
        direction[i % n_features] = 1.0
        direction[(i + 1) % n_features] = 0.3
        direction /= np.linalg.norm(direction)

        t = np.linspace(-5, 5, n)
        X_line = np.outer(t, direction)

        # Offset to pass through/near sphere
        offset = np.zeros(n_features)
        offset[(i + 2) % n_features] = (i - 0.5) * 2
        X_line += offset

        X_list.append(X_line)
        y_list.append(np.full(n, i, dtype=np.int64))

    # Generate sphere
    n_sphere = n_per_component[n_lines]
    X_sphere = rng.randn(n_sphere, n_features)
    X_sphere /= np.linalg.norm(X_sphere, axis=1, keepdims=True)
    X_sphere *= sphere_radius

    X_list.append(X_sphere)
    y_list.append(np.full(n_sphere, n_lines, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if noise > 0:
        X += rng.randn(*X.shape) * noise

    if shuffle:
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

    return validate_data(X, y)


def make_five_interlocking_circles(
    n_samples_per_circle: int = 100,
    radius: float = 2.0,
    noise: float = 0.0,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate five interlocking circles - Olympic rings (legacy toy problem #3).

    Exact reproduction of the original S3FC toy problem:
    - Circle 1 (blue): center=(0, 0)
    - Circle 2 (black): center=(3, 0)
    - Circle 3 (red): center=(6, 0)
    - Circle 4 (yellow): center=(1.5, -2)
    - Circle 5 (green): center=(4.5, -2)
    - All circles have radius=2
    - Points evenly spaced around each circle
    - Noise added to both x and y coordinates

    Parameters
    ----------
    n_samples_per_circle : int, default=100
        Number of points per circle (total = 5 * n_samples_per_circle).
    radius : float, default=2.0
        Radius of all circles.
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to both coordinates.
    random_state : int, RandomState, or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Circle labels (0-4).
    """
    rng = check_random_state(random_state)
    n = n_samples_per_circle

    # Circle centers (fixed positions matching original)
    # Top row: blue, black, red
    # Bottom row: yellow, green
    centers = [
        (0.0, 0.0),      # Circle 0 - blue
        (3.0, 0.0),      # Circle 1 - black
        (6.0, 0.0),      # Circle 2 - red
        (1.5, -2.0),     # Circle 3 - yellow
        (4.5, -2.0),     # Circle 4 - green
    ]

    X_list = []
    y_list = []

    for label, (cx, cy) in enumerate(centers):
        # Angles: 2*pi/n * i for i in 0..n-1 (matching original)
        angles = np.array([2 * np.pi / n * i for i in range(n)])
        x = np.cos(angles) * radius + cx
        y_coord = np.sin(angles) * radius + cy

        # Add noise to both coordinates
        if noise > 0:
            x = x + rng.normal(0, noise, n)
            y_coord = y_coord + rng.normal(0, noise, n)

        X_list.append(np.column_stack([x, y_coord]))
        y_list.append(np.full(n, label, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return validate_data(X, y)
