"""Toy datasets for the lectures, with a finance flavour.

Synthetic generators are small and reproducible so that figures in the notes
and slides regenerate identically.  They are *not* real market data.  For real
market data see ``load_returns()`` and ``load_basket_meta()``.
"""

from __future__ import annotations

import pathlib

import numpy as np


DEFAULT_SEED = 42

_DATA_DIR = pathlib.Path(__file__).parents[2] / "data" / "processed"


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


def correlated_returns(
    n: int = 500,
    rho: float = 0.6,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Two asset return series with a given *linear* (Pearson) correlation."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    return 0.01 * z[:, 0], 0.01 * z[:, 1]


def nonlinear_dependence(
    n: int = 500,
    noise: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """A strong non-linear dependence that Pearson almost completely misses.

    ``y = x**2 + noise`` over a symmetric range has near-zero linear
    correlation but is clearly (up to noise) deterministically dependent —
    the canonical motivation for mutual information / kernel measures.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    y = x**2 + noise * rng.standard_normal(size=n)
    return x, y


def two_banks(
    n: int = 500,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two stylised bank-stock return series with occasional joint crashes.

    Returns:
        ``(a, b, crash_mask)`` where ``crash_mask`` marks the ~10 % of days
        where both assets fell sharply together — the tail dependence that
        Pearson correlation misses.
    """
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, n)
    a = 0.80 * f + 0.40 * rng.normal(0, 1, n)
    b = 0.80 * f + 0.40 * rng.normal(0, 1, n)
    crash = rng.random(n) < 0.10
    shock = rng.uniform(1.8, 3.4, int(crash.sum()))
    a[crash] -= shock
    b[crash] -= shock
    return a, b, crash


def convex_payoff(
    n: int = 500,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """V-shaped (option-like) return pair: strong dependence, near-zero Pearson.

    ``v = |u| + noise`` — knowing ``u`` determines ``v`` almost exactly, yet
    Pearson reports ~0 because the relationship is symmetric around zero.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, n)
    v = np.abs(u) + 0.25 * rng.normal(0, 1, n)
    return u, v


def credit_scores_by_group(
    n: int = 600,
    gap: float = 0.8,
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray]:
    """Toy credit-scoring dataset with a sensitive attribute.

    The model score depends partly on the sensitive group, illustrating the
    fairness problem of *dependence between predictions and a protected
    variable*.

    Returns:
        Dict with ``score`` (model output), ``group`` (0/1 sensitive
        attribute), and ``repaid`` (ground-truth label).
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=n)
    latent = rng.standard_normal(size=n) + gap * group
    score = 1.0 / (1.0 + np.exp(-latent))
    repaid = (rng.uniform(size=n) < score).astype(int)
    return {"score": score, "group": group, "repaid": repaid}


def tail_dependent_pair(
    n: int = 2000,
    df: int = 3,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Two fat-tailed return series that crash together (standardized).

    A shared Student-*t* factor plus idiosyncratic *t* noise: the marginals are
    heavy-tailed and the pair shows strong *joint* lower-tail dependence — the
    co-crash that Gaussian Monte-Carlo understates and that Gaussianization,
    run backward, can reproduce.
    """
    rng = np.random.default_rng(seed)
    f = rng.standard_t(df, n)
    a = 0.8 * f + 0.4 * rng.standard_t(df, n)
    b = 0.8 * f + 0.4 * rng.standard_t(df, n)
    xy = np.column_stack([a, b])
    xy = (xy - xy.mean(0)) / xy.std(0)
    return xy[:, 0], xy[:, 1]


def fair_regression(
    n: int = 800,
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray]:
    """Regression data whose target depends on ``S`` both linearly *and* curvily.

    ``y = 0.6 x + 0.7 S + 0.7 (S^2 - 1) + noise``.  Linear neutrality removes the
    ``0.7 S`` term but leaves the ``S^2`` curve — the hole that the kernel (CKA)
    penalty is built to close.

    Returns:
        Dict with ``x`` (legitimate feature), ``s`` (sensitive variable), and
        ``y`` (target).
    """
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n)
    x = rng.standard_normal(n)
    y = 0.6 * x + 0.7 * s + 0.7 * (s**2 - 1.0) + 0.3 * rng.standard_normal(n)
    return {"x": x, "s": s, "y": y}


def heteroscedastic_gap(
    n: int = 60,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """A 1-D regression sample with a gap in the inputs.

    Training inputs cluster in two regions with empty space between (and beyond);
    the target is ``sin(1.3 x)`` plus mild noise.  Predictive bands should widen
    over the gap and the extrapolation tails — where the model has seen no data.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    x = np.concatenate([rng.uniform(-3, -1, half), rng.uniform(1, 3, n - half)])
    y = np.sin(1.3 * x) + 0.12 * rng.standard_normal(x.size)
    return x, y


def heteroscedastic_noise(
    n: int = 200,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """A 1-D regression sample whose noise level *grows with the input*.

    ``y = sin(1.3 x) + sigma(x) * eps`` with ``sigma(x)`` ramping from quiet on
    the left to loud on the right.  Unlike :func:`heteroscedastic_gap` (whose
    noise is *constant* and only the input density has a gap), here the
    aleatoric noise is itself a function of ``x`` — so a single global error bar
    is wrong everywhere, and the honest fix is to **predict the noise**
    ``sigma(x)`` from the residual magnitudes.

    Returns:
        ``(x, y)`` with ``x`` sorted and dense on ``[-3, 3]``.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3.0, 3.0, n))
    sigma = 0.05 + 0.45 * (x + 3.0) / 6.0  # ramps ~0.05 -> ~0.50 across the range
    y = np.sin(1.3 * x) + sigma * rng.standard_normal(x.size)
    return x, y


def info_blocks(
    n: int = 2000,
    dx: int = 2,
    dy: int = 2,
    within: float = 0.7,
    between: float = 0.5,
    fat_tails: bool = False,
    df: int = 3,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Two multidimensional blocks with known information-Venn region values.

    Builds ``x = [x_1, ..., x_dx]`` and ``y = [y_1, ..., y_dy]`` from a shared
    linear factor model so the Watanabe total-correlation decomposition has a
    *closed form*.  Each standardized component is::

        c_i = between·g + within·f_block + sqrt(1 - between² - within²)·eps_i

    with unit-variance factors: a global ``g`` shared by **all** components, a
    per-block factor ``f_x`` / ``f_y`` shared within a block, and idiosyncratic
    ``eps_i``.  The resulting covariance has unit diagonal, within-block
    off-diagonals ``between² + within²`` and cross-block off-diagonals
    ``between²``.

    The dials:

    * ``between=0`` makes the two blocks **independent** — ``I(x; y) = 0`` exactly
      (regardless of ``within``).
    * ``within`` raises the within-block correlation *above* its global-factor
      floor ``between²``, so it grows ``T(x)`` / ``T(y)``.  Because the global
      factor also couples components within a block, ``T(x) = T(y) = 0`` requires
      ``within = 0`` **and** ``between = 0`` (a single component, ``dx = 1``, also
      has ``T(x) = 0`` trivially).

    Args:
        n: number of samples (rows).
        dx: components in block ``x``.
        dy: components in block ``y``.
        within: per-block factor loading (raises ``T(x)``, ``T(y)``).
        between: global factor loading (raises ``I(x; y)``).
        fat_tails: if ``True`` the factors are standardized Student-*t*, adding
            heavy tails / tail dependence on top of the same covariance; the
            closed form is then a Gaussian/linear *reference*, not exact.
        df: degrees of freedom for the Student-*t* factors (``fat_tails=True``).
        seed: RNG seed.

    Returns:
        Dict with ``x`` ``(n, dx)``, ``y`` ``(n, dy)``, ``xy`` ``(n, dx + dy)``,
        the analytic covariance ``Sigma`` ``(dx + dy, dx + dy)``, a ``truth`` dict
        of region values **in nats** (``T_x``, ``T_y``, ``I_xy``, ``T_xy``,
        ``H_x``, ``H_y``, ``H_xy``), and ``truth_exact`` — ``True`` for the
        Gaussian model, ``False`` for ``fat_tails`` (where ``truth`` is the
        Gaussian reference and the heavy-tailed sample carries extra dependence
        above it).

    Raises:
        ValueError: if ``between² + within² > 1`` (no idiosyncratic variance
            left) or any of ``n``, ``dx``, ``dy`` is not positive.
    """
    if n < 1 or dx < 1 or dy < 1:
        raise ValueError("n, dx and dy must all be positive")
    shared = between**2 + within**2  # within-block off-diagonal correlation
    if shared > 1.0 + 1e-12:
        raise ValueError(
            f"between² + within² = {shared:.3f} > 1 — no idiosyncratic variance left"
        )

    rng = np.random.default_rng(seed)
    d = dx + dy
    idiosyncratic = float(np.sqrt(max(0.0, 1.0 - shared)))

    def draw(size):
        if fat_tails:
            scale = np.sqrt((df - 2) / df)  # standardize Student-t to unit variance
            return rng.standard_t(df, size=size) * scale
        return rng.standard_normal(size=size)

    g = draw(n)  # global factor → couples everything → I(x; y)
    f_x, f_y = draw(n), draw(n)  # per-block factors → T(x), T(y)
    eps = draw((n, d))  # idiosyncratic noise → unique entropy

    block_factor = np.empty((n, d))
    block_factor[:, :dx] = f_x[:, None]
    block_factor[:, dx:] = f_y[:, None]
    components = between * g[:, None] + within * block_factor + idiosyncratic * eps

    # Analytic covariance: unit diag, within-block between²+within², cross between².
    cov = np.full((d, d), float(between**2))
    cov[:dx, :dx] = shared
    cov[dx:, dx:] = shared
    np.fill_diagonal(cov, 1.0)

    # Closed-form Gaussian regions in nats (exact iff the factors are Gaussian).
    _, logdet = np.linalg.slogdet(cov)
    _, logdet_x = np.linalg.slogdet(cov[:dx, :dx])
    _, logdet_y = np.linalg.slogdet(cov[dx:, dx:])
    log_2pi_e = float(np.log(2.0 * np.pi * np.e))
    truth = {
        "T_x": -0.5 * logdet_x,  # ΣᵢH(xᵢ) - H(x) = -½ln|Σ_x|
        "T_y": -0.5 * logdet_y,
        "I_xy": 0.5 * (logdet_x + logdet_y - logdet),  # ½ln(|Σ_x||Σ_y|/|Σ|)
        "T_xy": -0.5 * logdet,  # -½ln|Σ|
        "H_x": 0.5 * (dx * log_2pi_e + logdet_x),  # ½ln((2πe)^dx |Σ_x|)
        "H_y": 0.5 * (dy * log_2pi_e + logdet_y),
        "H_xy": 0.5 * (d * log_2pi_e + logdet),
    }

    return {
        "x": components[:, :dx],
        "y": components[:, dx:],
        "xy": components,
        "Sigma": cov,
        "truth": truth,
        "truth_exact": not fat_tails,
    }


# ---------------------------------------------------------------------------
# Real-data loaders (require pandas — installed in the docs/demos environment)
# ---------------------------------------------------------------------------


def load_returns():  # -> pd.DataFrame
    """Load the 14-stock S&P 500 returns basket (5 years, daily).

    Returns a ``pd.DataFrame`` indexed by date with ticker symbols as columns.
    Requires ``pandas`` and the file ``data/processed/returns_basket.csv``.
    """
    import pandas as pd

    path = _DATA_DIR / "returns_basket.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_basket_meta():  # -> pd.DataFrame
    """Load metadata for the returns basket (ticker → label, sector).

    Returns a ``pd.DataFrame`` with columns ``ticker``, ``label``,
    ``sector``.
    """
    import pandas as pd

    path = _DATA_DIR / "basket_meta.csv"
    return pd.read_csv(path)


def load_credit():  # -> pd.DataFrame
    """Load the real income/fairness dataset (US Census ACS, via folktables).

    A protected attribute ``sex`` (1 = male), a binary ``income_high`` target
    (personal income over $50k), and real proxy features that leak the protected
    attribute even when ``sex`` is dropped — above all ``occupation`` (strongly
    sex-segregated), plus ``education_years``, ``hours_per_week``, ``marital``,
    and a near-neutral ``age`` control.  Sourced from the ACSIncome task on the
    2018 ACS (California, 1-year) by ``scripts/download_credit_data.py``.

    Requires ``pandas`` and ``data/processed/credit_acs.csv``.
    """
    import pandas as pd

    path = _DATA_DIR / "credit_acs.csv"
    return pd.read_csv(path)
