"""Gaussianization helpers.

The 1-D building block (rank → Gaussian) lives here.  Full, multivariate
Rotation-Based Iterative Gaussianization is provided by the author's
`rbig <https://github.com/jejjohnson/rbig>`_ package — import it directly::

    from rbig import AnnealedRBIG
    model = AnnealedRBIG(n_layers=30, rotation="pca", random_state=0).fit(X)
    Z = model.transform(X)        # forward: ~ standard Gaussian
    X_gen = model.sample(1000)     # backward: new samples like X

``rbig`` is NumPy/scikit-learn based, so it is installed in the light render and
test environments (the ``models`` group / ``docs`` feature), not only the heavy
``demos`` env.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def gaussianize_1d(v: np.ndarray) -> np.ndarray:
    """Map a 1-D array to a standard normal via the rank → Gaussian-quantile transform.

    Uses the Blom plotting-position ``(rank - 0.5) / n`` so the result stays
    finite (never exactly 0 or 1 before ``norm.ppf``).  Removes any monotone
    distortion of a single variable; it does **not** touch cross-dependence —
    that is what the rotation step of RBIG is for.
    """
    v = np.asarray(v, float)
    n = v.size
    ranks = rankdata(v)
    return norm.ppf((ranks - 0.5) / n)
