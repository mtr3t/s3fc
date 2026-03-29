#!/usr/bin/env python3
"""
S3FC Demo: Cluster two lines passing through a sphere.

This is the flagship problem from the paper. S3FC separates
1D lines from a 2D spherical shell where they intersect.

Expected output: NMI ~ 0.966
"""

from s3fc import S3FC, nmi_score
from s3fc.toy_generator import make_two_lines_and_sphere

# Generate synthetic data: 2 lines + 1 sphere (300 points, 3 clusters)
X, y_true = make_two_lines_and_sphere(
    n_samples_per_structure=100,
    noise=0.0,
    random_state=42,
)

print(f"Data: {X.shape[0]} points in {X.shape[1]}D, {len(set(y_true))} clusters")

# Run S3FC with best parameters for this dataset
model = S3FC(
    n_clusters=3,
    sigma=22.015,
    gamma=1.2,
    K=2,
    fusion="power",
    solver="SCS",
)
labels = model.fit_predict(X)

nmi = nmi_score(y_true, labels)
print(f"NMI: {nmi:.4f}")
