"""Tests for the penalized-neutrality helper (loss = fit + mu * Dep)."""

from __future__ import annotations

import numpy as np

from lectures.datasets import fair_regression
from lectures.measures import cka_rbf
from lectures.neutralize import fair_fit


def test_fair_fit_trades_accuracy_for_neutrality() -> None:
    """A larger mu lowers dependence on S (and does not improve the fit)."""
    data = fair_regression(n=500, seed=0)
    x, s, y = data["x"], data["s"], data["y"]

    p_plain = fair_fit(x, y, s, mu=0.0)
    p_fair = fair_fit(x, y, s, mu=40.0)

    dep_plain = cka_rbf(p_plain, s)
    dep_fair = cka_rbf(p_fair, s)
    assert dep_fair < dep_plain

    def accuracy(p: np.ndarray) -> float:
        return 1.0 - np.mean((p - y) ** 2) / np.var(y)

    assert accuracy(p_fair) <= accuracy(p_plain) + 1e-6


def test_fair_fit_plain_is_reasonable_regression() -> None:
    """At mu=0 the plain fit explains a good share of the variance."""
    data = fair_regression(n=500, seed=1)
    p = fair_fit(data["x"], data["y"], data["s"], mu=0.0)
    r2 = 1.0 - np.mean((p - data["y"]) ** 2) / np.var(data["y"])
    assert r2 > 0.6
