"""Dependence measures for the lecture ladder.

Covers the full ladder taught in L1 — from the cheap linear tools up to
kernel-based and information-theoretic measures.  The first three are
dependency-free (numpy/scipy only).  The RBIG-based mutual-information
estimator lazily imports ``rbig`` so the module stays importable in
environments where that package is absent.

Ladder summary
--------------
Pearson         straight-line co-movement only
Spearman        any monotonic relationship (rank-based)
distance corr   any dependence (zero *iff* independent)
MI (discrete)   mutual information from a joint probability table
Linfoot R       MI rescaled onto the [-1, 1] correlation axis
HSIC            kernel independence test without estimating a density
CKA             normalized HSIC — bounded [0,1], used as L2 penalty
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


# Kernel methods build a dense n x n Gram matrix (several of them).  Guard
# against accidentally allocating gigabytes: at n = 5000 each matrix is ~200 MB,
# and HSIC/CKA build a handful.  Subsample (or mini-batch) above this.
_MAX_KERNEL_N = 5000


def _check_kernel_size(n: int) -> None:
    if n > _MAX_KERNEL_N:
        raise ValueError(
            f"kernel measure called with n={n}; it builds dense {n}x{n} "
            f"matrices (O(n^2) memory/time). Subsample to <= {_MAX_KERNEL_N} "
            "rows first (e.g. df.sample(2000))."
        )


# ---------------------------------------------------------------------------
# No-density road: compare(transform(x), transform(y))
# ---------------------------------------------------------------------------


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Sample distance correlation between two 1-D arrays.

    Zero *only* when the variables are genuinely independent; grows with
    curved or tail-driven relationships.  Range [0, 1].

    Implementation follows the definition directly (no extra dependency).
    """
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float).reshape(-1, 1)
    a = np.abs(x - x.T)
    b = np.abs(y - y.T)
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    denom = np.sqrt(dvar_x * dvar_y)
    return float(np.sqrt(max(dcov2, 0.0) / denom)) if denom > 0 else 0.0


def measure_trio(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return Pearson, Spearman, and distance correlation for one pair.

    The three-number summary used in the L1 ladder: they should *disagree*
    whenever the relationship is nonlinear.
    """
    return {
        "pearson": float(pearsonr(x, y)[0]),
        "spearman": float(spearmanr(x, y)[0]),
        "distance": distance_correlation(x, y),
    }


# ---------------------------------------------------------------------------
# Through-the-density road: information-theoretic measures
# ---------------------------------------------------------------------------


def mutual_information_bits(joint: np.ndarray) -> float:
    """Mutual information in *bits* from a (normalised) joint probability table.

    Args:
        joint: 2-D array whose entries are non-negative and sum to 1 (or
               will be normalised).  Rows = values of X, cols = values of Y.

    Returns:
        I(X; Y) in bits.  Zero when X and Y are independent.
    """
    P = np.asarray(joint, float)
    P = P / P.sum()
    px = P.sum(1, keepdims=True)
    py = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0, P * np.log2(P / (px @ py)), 0.0)
    return float(terms.sum())


def linfoot_r(mi_nats: float) -> float:
    """Rescale mutual information (in nats) onto the Pearson correlation scale.

    ``R = sqrt(1 - exp(-2 * MI))``

    For a bivariate Gaussian this returns exactly ``|rho|``, so any gap
    between Linfoot R and Pearson's r on real data is the non-Gaussian,
    tail-driven dependence that correlation structurally cannot represent.
    """
    return float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * mi_nats))))


def nats_to_bits(nats: float) -> float:
    """Convert an information quantity from nats to bits."""
    return nats / np.log(2.0)


# ---------------------------------------------------------------------------
# Kernel independence test (no density required)
# ---------------------------------------------------------------------------


def hsic_rbf(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float | None = None,
) -> float:
    """Hilbert-Schmidt Independence Criterion with RBF kernels.

    Zero (with a characteristic kernel) only under independence.  Does not
    require a density estimate; scales to many variables.

    Args:
        x: 1-D array of length n.
        y: 1-D array of length n.
        sigma: RBF bandwidth.  If ``None`` the median heuristic is used.

    Returns:
        HSIC estimate (non-negative).
    """
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float).reshape(-1, 1)
    n = len(x)
    _check_kernel_size(n)

    def rbf(a: np.ndarray) -> np.ndarray:
        d2 = (a - a.T) ** 2
        s = sigma or float(np.sqrt(np.median(d2[d2 > 0]) / 2))
        return np.exp(-d2 / (2.0 * s * s))

    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(rbf(x) @ H @ rbf(y) @ H) / (n - 1) ** 2)


# ---------------------------------------------------------------------------
# Normalized kernel dependence (the L2 fairness/neutrality penalty)
# ---------------------------------------------------------------------------


def cka_rbf(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Centered Kernel Alignment with RBF kernels — normalized HSIC in ``[0, 1]``.

    ``CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))``.  Like Pearson
    normalizes covariance, CKA normalizes HSIC: 0 under independence, 1 when the
    two are deterministically related.  Because it is bounded, it makes the
    fairness weight ``mu`` interpretable and the accuracy/neutrality trade-off
    axes comparable.  This is the hero ``Dep`` of the Fair Learning chapter.
    """
    hxy = hsic_rbf(x, y, sigma)
    hxx = hsic_rbf(x, x, sigma)
    hyy = hsic_rbf(y, y, sigma)
    denom = np.sqrt(hxx * hyy)
    return float(hxy / denom) if denom > 0 else 0.0


def cka_linear(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA — normalized HSIC with linear kernels.

    The cheap, linear-only rung: it sees straight-line dependence and is blind
    to curves (for 1-D inputs it equals the squared Pearson correlation).  The
    gap to :func:`cka_rbf` is exactly the nonlinear dependence linear neutrality
    leaves behind.
    """
    x = np.asarray(x, float).reshape(len(x), -1)
    y = np.asarray(y, float).reshape(len(y), -1)
    n = len(x)
    _check_kernel_size(n)
    H = np.eye(n) - np.ones((n, n)) / n
    kx = x @ x.T
    ky = y @ y.T

    def hsic(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.trace(a @ H @ b @ H))

    denom = np.sqrt(hsic(kx, kx) * hsic(ky, ky))
    return float(hsic(kx, ky) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Tail co-movement and joint mutual information
# ---------------------------------------------------------------------------


def cocrash_fraction(x: np.ndarray, y: np.ndarray, q: float = 0.10) -> float:
    """Fraction of ``x``'s lower-``q`` tail days on which ``y`` also crashes.

    Estimates ``P(Y in lower-q tail | X in lower-q tail)`` — the joint tail
    dependence a diversified portfolio cares about and that Gaussian models
    structurally understate.  Independent series give ~``q``; co-crashing pairs
    give much more.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    tx, ty = np.quantile(x, q), np.quantile(y, q)
    left = x <= tx
    if not left.any():
        return 0.0
    return float((y[left] <= ty).mean())


def mutual_information_xy(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Mutual information (in *nats*) of a continuous pair, via a 2-D histogram.

    A quick joint estimator that, unlike a correlation, lights up for curved and
    tail dependence.  Used to show that *joint* Gaussianization drives the shared
    information toward zero where per-marginal Gaussianization cannot.
    """
    P = np.histogram2d(np.asarray(x, float), np.asarray(y, float), bins=bins)[0]
    return mutual_information_bits(P) * np.log(2.0)
