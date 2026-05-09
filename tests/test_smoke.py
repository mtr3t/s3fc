"""Smoke test: package imports cleanly and the flagship demo
runs above the published NMI threshold.

Reproduces the headline result from Radice & Phillips (2026), FLAIRS-39:
two lines passing through a sphere shell, n=300, NMI ~ 0.966.
"""
from s3fc import S3FC, nmi_score
from s3fc.toy_generator import make_two_lines_and_sphere


def test_flagship_nmi_above_threshold():
    X, y = make_two_lines_and_sphere(
        n_samples_per_structure=100,
        noise=0.0,
        random_state=42,
    )
    model = S3FC(
        n_clusters=3,
        sigma=22.015,
        gamma=1.2,
        K=2,
        fusion="power",
        solver="SCS",
    )
    pred = model.fit_predict(X)
    nmi = nmi_score(y, pred)
    assert nmi > 0.95, f"Flagship NMI regression: got {nmi:.4f}, expected > 0.95"
    return nmi


if __name__ == "__main__":
    nmi = test_flagship_nmi_above_threshold()
    print(f"Smoke test passed. NMI = {nmi:.4f}")
