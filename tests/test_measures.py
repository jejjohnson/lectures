"""Tests for the dependence measures added for the lecture chapters."""

from __future__ import annotations

import numpy as np
import pytest

from lectures.measures import (
    cka_linear,
    cka_rbf,
    cocrash_fraction,
    mutual_information_xy,
)


def test_cka_rbf_bounds_and_extremes() -> None:
    """CKA is in [0, 1]: ~0 for independent, near 1 for identical."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    y = rng.standard_normal(400)
    indep = cka_rbf(x, y)
    same = cka_rbf(x, x)
    assert 0.0 <= indep <= 1.0
    assert 0.0 <= same <= 1.0
    assert indep < 0.1
    assert same > 0.9


def test_cka_rbf_sees_nonlinear_dependence() -> None:
    """RBF CKA catches a curve that linear CKA is blind to."""
    rng = np.random.default_rng(1)
    s = rng.standard_normal(500)
    y = s**2 + 0.1 * rng.standard_normal(500)
    assert cka_rbf(y, s) > cka_linear(y, s)
    assert cka_linear(y, s) < 0.1  # symmetric quadratic -> ~zero linear dependence


def test_cocrash_fraction() -> None:
    """Co-crash fraction is ~q for independent series, high for a coupled pair."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4000)
    y = rng.standard_normal(4000)
    assert abs(cocrash_fraction(x, y, q=0.1) - 0.1) < 0.05
    z = x + 0.05 * rng.standard_normal(4000)
    assert cocrash_fraction(x, z, q=0.1) > 0.7


def test_kernel_size_guard_raises() -> None:
    """Kernel measures refuse oversized inputs instead of allocating gigabytes."""
    big = np.zeros(6000)
    with pytest.raises(ValueError, match=r"subsample|kernel"):
        cka_rbf(big, big)
    with pytest.raises(ValueError, match=r"subsample|kernel"):
        cka_linear(big, big)


def test_mutual_information_xy_nonnegative_and_curved() -> None:
    """Histogram MI is ~0 for independent data and positive for a curve."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(3000)
    y = rng.standard_normal(3000)
    s = rng.standard_normal(3000)
    curved = s**2 + 0.1 * rng.standard_normal(3000)
    assert mutual_information_xy(x, y) < mutual_information_xy(s, curved)
    assert mutual_information_xy(x, y) >= 0.0
