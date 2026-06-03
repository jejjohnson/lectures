# %% [markdown]
# # Demo 1 — Measuring dependence
# Pearson vs. Spearman vs. a mutual-information estimate on return-like data.
# Jupytext percent-format; run `pixi run notebooks-to-ipynb` to get an .ipynb.

# %%
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path.cwd() / "src"))

import numpy as np
from scipy.stats import spearmanr

from lectures import datasets, plotting

plotting.set_style("notes")

# %%
x, y = datasets.correlated_returns(rho=0.6)
print("Pearson :", round(float(np.corrcoef(x, y)[0, 1]), 3))
print("Spearman:", round(float(spearmanr(x, y).statistic), 3))

# %% [markdown]
# Next: swap in real returns, then estimate mutual information with `rbig`.
