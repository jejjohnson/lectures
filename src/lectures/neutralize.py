"""Penalized neutrality: ``loss = fit(y_hat, y) + mu * Dep(y_hat, S)``.

The spine of the Fair Learning chapter in one function.  Once a dependence can be
*measured* (chapter 1), it can be *penalized*: a model fits the target while a
dependence term pulls its predictions off a sensitive / nuisance variable ``S``.
Sweeping ``mu`` from 0 (pure accuracy) upward traces the accuracy-vs-neutrality
trade-off curve — the central artifact of the chapter.

The optimum of that penalized objective removes the part of the prediction that
``S`` can explain, so we realize it deterministically: fit a plain model, then
*orthogonalize* the predictions against a feature map of ``S`` (polynomial, so it
removes curves as well as lines), with a ridge whose strength is set by ``mu``.
This is a teaching-sized, NumPy-only stand-in for the Keras ``fairkl``
``FairModelWrapper``; for real fair kernel learning use
`keras-fairkl <https://github.com/jejjohnson/keras-fairkl>`_ (the ``demos`` env).
"""

from __future__ import annotations

import numpy as np


def _design_matrix(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Feature basis rich enough to capture linear *and* curved S-dependence."""
    x = np.asarray(x, float)
    s = np.asarray(s, float)
    return np.column_stack([np.ones_like(x), x, s, s**2 - 1.0, x * s])


def _s_basis(s: np.ndarray, degree: int = 3) -> np.ndarray:
    """Centered, unit-scaled polynomial features of ``S`` (the part to remove)."""
    s = np.asarray(s, float)
    cols = []
    for k in range(1, degree + 1):
        c = s**k
        c = c - c.mean()
        sd = c.std() or 1.0
        cols.append(c / sd)
    return np.column_stack(cols)


def fair_fit(
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    mu: float,
    degree: int = 3,
) -> np.ndarray:
    """Fit ``y`` from ``x`` while penalizing dependence on ``s``; return predictions.

    At ``mu = 0`` this is the plain least-squares model (which, given the basis,
    can track ``s`` strongly).  As ``mu`` grows, a ridge-orthogonalization
    against a polynomial feature map of ``s`` removes the ``s``-explainable
    component of the predictions — driving the kernel dependence
    (:func:`~lectures.measures.cka_rbf`) toward zero and trading away the
    accuracy that rode on ``s``.

    Args:
        x: 1-D feature array.
        y: 1-D regression target.
        s: 1-D sensitive / nuisance variable to be neutralized.
        mu: neutrality weight (0 = pure fit; larger = more neutral).
        degree: highest polynomial power of ``s`` removed (3 covers the curve).

    Returns:
        Predictions on the training rows.
    """
    y = np.asarray(y, float)
    phi = _design_matrix(x, s)
    theta, *_ = np.linalg.lstsq(phi, y, rcond=None)
    y_hat = phi @ theta
    if mu <= 0:
        return y_hat

    b = _s_basis(s, degree)
    # Ridge removal scaled to the gram magnitude (~n per standardized column):
    # lambda = n / mu sweeps from light (small mu) to near-complete (large mu).
    lam = b.shape[0] / mu
    gram = b.T @ b + lam * np.eye(b.shape[1])
    coef = np.linalg.solve(gram, b.T @ y_hat)
    return y_hat - b @ coef
