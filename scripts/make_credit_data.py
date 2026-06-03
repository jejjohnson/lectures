"""Generate an Adult-census-*like* credit dataset, bundled offline.

Real OpenML/UCI mirrors of the Adult census data are network-blocked in this
environment, so we synthesise a dataset that reproduces the *structure* the
fairness lesson needs: a protected attribute (``sex``) the target depends on,
plus correlated proxy features through which that dependence **leaks** even when
``sex`` is dropped from the model. The sample is male-skewed (~67% male) with a
sizeable high-income gap by sex, mirroring Adult's gross structure. With the
default seed it realises ~46% high-income overall. This is NOT the real Adult
data.

Run:  python scripts/make_credit_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def make(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Build the synthetic credit table with leaky proxy features."""
    rng = np.random.default_rng(seed)
    sex = rng.binomial(1, 0.67, n)  # 1 = Male
    age = np.clip(rng.normal(39, 13, n), 17, 90)
    # proxies STRONGLY correlated with sex (the leakage channels)
    education = np.clip(rng.normal(10, 2.5, n) + 1.6 * sex, 1, 16)
    hours = np.clip(rng.normal(36, 11, n) + 7 * sex, 1, 99)
    occ_prestige = np.clip(rng.normal(0.40, 0.18, n) + 0.16 * sex, 0, 1)
    # high-income propensity: legit features + a direct unfair sex effect
    z = (
        -7.5
        + 0.04 * age
        + 0.28 * education
        + 0.03 * hours
        + 2.0 * occ_prestige
        + 0.5 * sex
        + rng.normal(0, 1.3, n)
    )
    p = 1 / (1 + np.exp(-z))
    income = (rng.random(n) < p).astype(int)
    return pd.DataFrame(
        {
            "age": age.round(0).astype(int),
            "education_num": education.round(0).astype(int),
            "hours_per_week": hours.round(0).astype(int),
            "occupation_prestige": occ_prestige.round(3),
            "sex": sex,
            "income_high": income,
        }
    )


def main() -> None:
    """Generate the dataset and report the gap + proxy correlations."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = make()
    out = PROCESSED / "adult_like.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  shape={df.shape}  P(high)={df.income_high.mean():.2f}")
    p_male = df[df.sex == 1].income_high.mean()
    p_female = df[df.sex == 0].income_high.mean()
    print(f"P(high|Male)={p_male:.2f}  P(high|Female)={p_female:.2f}")
    for c in ["education_num", "hours_per_week", "occupation_prestige"]:
        print(f"corr({c}, sex) = {df[c].corr(df.sex):.2f}")


if __name__ == "__main__":
    main()
