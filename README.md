# S3FC: Scalable Sparse Spectral Fusion Clustering

Multi-manifold clustering by fusing sparse subspace affinity with spectral affinity.

## Setup

Tested with Python 3.10 and CVXPY 1.7.5. Other versions may not work.

```bash
git clone https://github.com/mtr3t/s3fc.git
cd s3fc
conda create -n s3fc python=3.10 -y
conda activate s3fc
pip install -e .
pip install cvxpy==1.7.5
```

## Run

```bash
python examples/demo.py
```

Expected output:

```
Data: 300 points in 3D, 3 clusters
NMI: 0.9660
```

## What it does

S3FC clusters data that lies on multiple manifolds of mixed dimension (e.g., lines passing through a sphere). It builds two affinity matrices -- one from sparse self-representation, one from an RBF kernel -- and fuses them so that connections survive only where both views agree.

## Citation

If you use this code, please cite:

```
Radice, M. T. and Phillips, J. L. (2026).
S3FC: Scalable Sparse Spectral Fusion Clustering for Multi-Manifold Data.
In Proceedings of the 39th International FLAIRS Conference (FLAIRS-39).
```
