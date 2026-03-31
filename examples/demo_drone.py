#!/usr/bin/env python3
"""
S3FC Demo: Cluster real-world drone GPS data.

Three drones flew different patterns: two flew linear inspection
passes while a third performed an orbital loiter. S3FC separates
the flight paths perfectly.

Expected output: NMI = 1.0000
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from s3fc import S3FC, nmi_score


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


# Load drone GPS data
data_path = Path(__file__).parent.parent / "data" / "two_lines_and_sphere_drone_100.mat"
mat = loadmat(str(data_path))
X = mat["X"].astype(np.float64)
y_true = mat["labels_true"].flatten().astype(int)
# Ensure 0-indexed labels
if y_true.min() == 1:
    y_true = y_true - 1

print(f"Data: {X.shape[0]} points in {X.shape[1]}D, {len(set(y_true))} clusters")

# Run S3FC with best parameters for drone data
model = S3FC(
    n_clusters=3,
    sigma=0.05,
    gamma=0.1,
    K=0,
    fusion="power",
    solver="SCS",
)
labels = model.fit_predict(X)
labels = match_labels(y_true, labels)

nmi = nmi_score(y_true, labels)
print(f"NMI: {nmi:.4f}")

# Plot ground truth vs S3FC result
colors = ["#e41a1c", "#377eb8", "#4daf4a"]
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
    ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
