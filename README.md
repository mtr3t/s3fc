# S3FC: Scalable Sparse Spectral Fusion Clustering

[![CI](https://github.com/mtr3t/s3fc/actions/workflows/ci.yml/badge.svg)](https://github.com/mtr3t/s3fc/actions/workflows/ci.yml)

Multi-manifold clustering by fusing sparse subspace affinity with spectral affinity.

## Setup

Tested with Python 3.10 and CVXPY 1.7.5. Other versions may not work.

```bash
git clone https://github.com/mtr3t/s3fc.git
cd s3fc
conda create -n s3fc python=3.10 -y
conda activate s3fc
pip install -e .
```

All dependencies (including pinned versions) are handled by `pyproject.toml`.

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

## Datasets

### Bundled in this repo (`data/`)
- **Synthetic flagship** (`two_lines_and_sphere.mat`) -- 300 points, two 1D lines through a 2D sphere shell. Reproduces the paper's headline NMI = 0.966 via `examples/demo.py`.
- **Drone GPS demo** (`two_lines_and_sphere_drone_100.mat`) -- 300 points, real drone telemetry (two flight paths through an orbital loiter). Reproduces NMI = 1.000 via `examples/demo_drone.py`.

Synthetic dataset generators are in `src/s3fc/toy_generator.py`.

### Referenced in the paper, not bundled
The full paper benchmarks against several real-world datasets that are not redistributed here. To reproduce those results, download separately:
- **ORL Faces** -- https://cam-orl.co.uk/facedatabase.html
- **Yale B (yaleb32)** -- standard Cropped Yale B mirror (e.g., the dataset distributed with the SSC reference implementations)
- **Olivetti** -- `sklearn.datasets.fetch_olivetti_faces()`
- **Digits** -- `sklearn.datasets.load_digits()`

## Troubleshooting

### CVXPY version
S3FC was validated with **CVXPY 1.7.5** for the FLAIRS-39 paper.

- CVXPY 1.7.3 has an assertion bug that crashes solvers mid-run -- confirmed during development. Avoid.
- CVXPY 1.8.x (currently 1.8.2) was released after the paper and has not been validated against S3FC.

`pyproject.toml` pins `cvxpy==1.7.5` so a fresh `pip install -e .` gives the validated version. If a different CVXPY is already in your environment, the package will warn on import. To fix:

```bash
pip install cvxpy==1.7.5 --force-reinstall
```

### Windows / Intel: JAX solver discovery warning
On Windows you may see:

```
Encountered unexpected exception importing solver CUCLARABEL: UnicodeDecodeError
```

This is cosmetic -- CVXPY catches the error and falls back to SCS/CLARABEL/OSQP normally. Results are unaffected. To suppress, set `JAX_PLATFORMS=cpu` before running Python.

### Parallel execution (`n_jobs`)
- **macOS:** `n_jobs=-1` works (fork-based multiprocessing).
- **Windows:** `n_jobs=-1` can segfault under spawn-based multiprocessing. Use an explicit value (`n_jobs=8` or `n_jobs=16`).
- **Linux:** usually fine with `n_jobs=-1`, but explicit values are safer for reproducibility.

## Paper

S3FC was accepted to the **39th International Florida Artificial Intelligence Research Society Conference (FLAIRS-39)**, May 17–20, 2026, Marco Island, FL.

- Conference: https://www.flairs-39.info/
- Proceedings: forthcoming (link will be added when published)

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{radice2026s3fc,
  author    = {Radice, Matthew T. and Phillips, Joshua L.},
  title     = {S3FC: Scalable Sparse Spectral Fusion Clustering for Multi-Manifold Data},
  booktitle = {Proceedings of the 39th International FLAIRS Conference},
  year      = {2026},
  series    = {FLAIRS-39},
  address   = {Marco Island, FL, USA},
  month     = {May}
}
```

## License

Released under the [MIT License](LICENSE).
