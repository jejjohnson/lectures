"""Tests for the 1-D Gaussianization helper and rbig integration.

``gaussianize_1d`` is owned by this repo; the multivariate RBIG comes from the
author's ``rbig`` package.  These are integration smoke tests for the behaviour
the lecture figures rely on (decorrelation, co-crash-preserving sampling),
not a re-test of rbig's internals.
"""

from __future__ import annotations

import numpy as np
from rbig import AnnealedRBIG

from lectures.datasets import tail_dependent_pair
from lectures.gaussianization import gaussianize_1d
from lectures.measures import cocrash_fraction


def test_gaussianize_1d_is_standard_normal() -> None:
    """A skewed variable becomes ~standard normal after the rank transform."""
    rng = np.random.default_rng(0)
    v = rng.exponential(size=2000)
    g = gaussianize_1d(v)
    assert abs(g.mean()) < 0.1
    assert abs(g.std() - 1.0) < 0.1


def test_rbig_forward_decorrelates() -> None:
    """The forward map turns a correlated pair into ~uncorrelated marginals."""
    a, b = tail_dependent_pair(n=1500, seed=1)
    x = np.column_stack([a, b])
    z = np.asarray(AnnealedRBIG(n_layers=20, random_state=0).fit_transform(x))
    corr_before = abs(np.corrcoef(a, b)[0, 1])
    corr_after = abs(np.corrcoef(z[:, 0], z[:, 1])[0, 1])
    assert corr_after < corr_before
    assert corr_after < 0.15


def test_rbig_forward_marginals_are_gaussian() -> None:
    """The forward map produces ~standard-normal marginals (the spherical panel)."""
    a, b = tail_dependent_pair(n=1500, seed=2)
    x = np.column_stack([a, b])
    z = np.asarray(AnnealedRBIG(n_layers=20, random_state=0).fit_transform(x))
    assert np.allclose(z.mean(0), 0.0, atol=0.1)
    assert np.allclose(z.std(0), 1.0, atol=0.2)


def test_rbig_sample_reproduces_cocrash() -> None:
    """Samples drawn backward keep the joint tail dependence of the data."""
    a, b = tail_dependent_pair(n=2000, seed=3)
    x = np.column_stack([a, b])
    model = AnnealedRBIG(n_layers=30, random_state=0).fit(x)
    gen = np.asarray(model.sample(2000))
    cc_real = cocrash_fraction(a, b, q=0.1)
    cc_gen = cocrash_fraction(gen[:, 0], gen[:, 1], q=0.1)
    assert cc_real > 0.3
    assert abs(cc_gen - cc_real) < 0.15
