"""Tests for the synthetic dataset generators.

Focus: ``info_blocks`` — the multidimensional information-Venn generator whose
``truth`` dict must match the closed-form Gaussian region values.
"""

from __future__ import annotations

import numpy as np
import pytest

from lectures.datasets import info_blocks


def _regions_from_cov(cov: np.ndarray, dx: int, dy: int) -> dict[str, float]:
    """Recompute the Venn regions (nats) from a covariance matrix."""
    _, logdet = np.linalg.slogdet(cov)
    _, logdet_x = np.linalg.slogdet(cov[:dx, :dx])
    _, logdet_y = np.linalg.slogdet(cov[dx:, dx:])
    return {
        "T_x": -0.5 * logdet_x,
        "T_y": -0.5 * logdet_y,
        "I_xy": 0.5 * (logdet_x + logdet_y - logdet),
        "T_xy": -0.5 * logdet,
    }


def test_info_blocks_shapes_and_keys() -> None:
    """Shapes line up with (dx, dy) and the truth dict is complete."""
    b = info_blocks(n=1000, dx=3, dy=2, seed=0)
    assert b["x"].shape == (1000, 3)
    assert b["y"].shape == (1000, 2)
    assert b["xy"].shape == (1000, 5)
    assert b["Sigma"].shape == (5, 5)
    assert set(b["truth"]) == {"T_x", "T_y", "I_xy", "T_xy", "H_x", "H_y", "H_xy"}
    assert b["truth_exact"] is True


def test_info_blocks_covariance_structure() -> None:
    """Sigma has unit diagonal, within-block = between²+within², cross = between²."""
    within, between, dx, dy = 0.7, 0.5, 2, 2
    cov = info_blocks(within=within, between=between, dx=dx, dy=dy)["Sigma"]
    assert np.allclose(np.diag(cov), 1.0)
    assert cov[0, 1] == pytest.approx(between**2 + within**2)  # within block x
    assert cov[dx, dx + 1] == pytest.approx(between**2 + within**2)  # within block y
    assert cov[0, dx] == pytest.approx(between**2)  # cross block


def test_info_blocks_gaussian_truth_matches_samples() -> None:
    """Closed-form truth matches regions read off the empirical covariance."""
    b = info_blocks(n=200_000, dx=2, dy=2, within=0.7, between=0.5, seed=1)
    empirical = _regions_from_cov(np.cov(b["xy"], rowvar=False), 2, 2)
    for key, value in empirical.items():
        assert value == pytest.approx(b["truth"][key], abs=0.02)


def test_info_blocks_decomposition_identity() -> None:
    """The non-negative decomposition T_xy = T_x + T_y + I_xy holds exactly."""
    t = info_blocks(dx=3, dy=2, within=0.6, between=0.4)["truth"]
    assert t["T_xy"] == pytest.approx(t["T_x"] + t["T_y"] + t["I_xy"])
    for region in ("T_x", "T_y", "I_xy", "T_xy"):
        assert t[region] >= -1e-12  # all regions non-negative


def test_info_blocks_dials() -> None:
    """between drives I(x;y); within drives T(x); the clean zeros are exact."""
    # between = 0  ->  the two blocks are independent  ->  I_xy = 0 exactly
    assert info_blocks(between=0.0, within=0.7)["truth"]["I_xy"] == pytest.approx(0.0)
    # within = 0 AND between = 0  ->  everything independent  ->  T_x = T_y = 0
    indep = info_blocks(between=0.0, within=0.0)["truth"]
    assert indep["T_x"] == pytest.approx(0.0)
    assert indep["T_y"] == pytest.approx(0.0)
    # the dials are monotone: more within -> more T_x; more between -> more I_xy
    assert (
        info_blocks(within=0.8, between=0.3)["truth"]["T_x"]
        > info_blocks(within=0.3, between=0.3)["truth"]["T_x"]
    )
    assert (
        info_blocks(within=0.4, between=0.7)["truth"]["I_xy"]
        > info_blocks(within=0.4, between=0.3)["truth"]["I_xy"]
    )


def test_info_blocks_scalar_mi_recovers_gaussian() -> None:
    """With dx = dy = 1 the cross MI is the scalar Gaussian MI at rho = between²."""
    between = 0.5
    mi = info_blocks(dx=1, dy=1, within=0.0, between=between)["truth"]["I_xy"]
    rho = between**2  # cross-block correlation when there is no within structure
    assert mi == pytest.approx(-0.5 * np.log(1.0 - rho**2))


def test_info_blocks_fat_tails_flag_and_finiteness() -> None:
    """Fat-tail mode marks truth inexact and still returns finite samples."""
    b = info_blocks(fat_tails=True, df=3, seed=4)
    assert b["truth_exact"] is False
    assert np.isfinite(b["xy"]).all()


def test_info_blocks_rejects_impossible_loadings() -> None:
    """between² + within² > 1 leaves no idiosyncratic variance — a ValueError."""
    with pytest.raises(ValueError):
        info_blocks(within=0.9, between=0.9)
