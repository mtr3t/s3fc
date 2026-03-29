#!/usr/bin/env python3
"""
S3FC Demo: Cluster two lines passing through a sphere.

This is the flagship problem from the paper. S3FC separates
1D lines from a 2D spherical shell where they intersect.

Expected output: NMI ~ 0.966
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from s3fc import S3FC, nmi_score
from s3fc.toy_generator import make_two_lines_and_sphere


def match_labels(y_true, y_pred):
    """Relabel y_pred so colors match y_true (Hungarian algorithm)."""
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost = np.zeros((len(labels_true), len(labels_pred)))
    for i, lt in enumerate(labels_true):
        for j, lp in enumerate(labels_pred):
            cost[i, j] = -np.sum((y_true == lt) & (y_pred == lp))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {labels_pred[c]: labels_true[r] for r, c in zip(row_ind, col_ind)}
    return np.array([mapping[l] for l in y_pred])


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
labels = match_labels(y_true, labels)

nmi = nmi_score(y_true, labels)
print(f"NMI: {nmi:.4f}")

# Plot ground truth vs S3FC result
colors = ["#e41a1c", "#377eb8", "#4daf4a"]  # red, blue, green
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "3d"})

for ax, c, title in [(ax1, y_true, "Ground Truth"),
                      (ax2, labels, f"S3FC (NMI = {nmi:.4f})")]:
    for k in np.unique(c):
        mask = c == k
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                   c=colors[k], s=15, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
