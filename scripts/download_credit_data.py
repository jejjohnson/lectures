"""Download and prepare the real income/fairness dataset the course runs on.

Source: the US Census **American Community Survey** (ACS) Public Use Microdata,
via the [`folktables`](https://github.com/socialfoundations/folktables) package
and its **ACSIncome** task (Ding et al., 2021 — the modern successor to UCI
Adult).  We take one large, diverse state (California, 2018 1-year), apply the
standard ACSIncome row filter, decode the coded columns into readable features,
and sample a manageable panel.

The target is ``income_high`` = personal income over $50k.  The protected
attribute is ``sex``.  The teaching point lives in ``occupation``: occupations are
strongly sex-segregated, so a model that keeps occupation after dropping sex can
*reconstruct* the protected attribute — real proxy leakage, not a synthetic one.

Run:  python scripts/download_credit_data.py
      (requires ``folktables`` + ``scikit-learn``; install with
      ``uv run --with folktables --with scikit-learn`` or add them to your env)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from folktables import ACSDataSource


STATES = ["CA"]
YEAR = "2018"
N_SAMPLE = 30_000
SEED = 0
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

# Census OCCP code ranges (2018) collapsed to nine readable occupation groups.
_OCC_RANGES: list[tuple[int, int, str]] = [
    (10, 960, "Management/Business/Finance"),
    (1005, 1980, "Computing/Engineering/Science"),
    (2001, 2920, "Education/Legal/Arts/Social"),
    (3000, 3655, "Healthcare"),
    (3700, 4655, "Service (food/clean/care/protective)"),
    (4700, 4965, "Sales"),
    (5000, 5940, "Office & Admin"),
    (6005, 9760, "Blue-collar (build/repair/prod/transport)"),
]
_MARITAL = {
    1: "Married",
    2: "Widowed",
    3: "Divorced",
    4: "Separated",
    5: "Never married",
}
# ACS SCHL attainment code -> approximate years of education.
_SCHL_YEARS = {16: 12, 17: 12, 18: 13, 19: 14, 20: 14, 21: 16, 22: 18, 23: 19, 24: 21}


def _occupation_group(code: float) -> str:
    """Map an OCCP occupation code to one of nine readable groups."""
    c = int(code)
    for lo, hi, name in _OCC_RANGES:
        if lo <= c <= hi:
            return name
    return "Military/Other"


def _schl_years(code: float) -> int:
    """Map an SCHL attainment code to approximate years of education."""
    s = int(code)
    return _SCHL_YEARS.get(s, min(max(s - 3, 0), 11))


def _acsincome_filter(df: pd.DataFrame) -> pd.DataFrame:
    """The standard ACSIncome row filter (working-age, employed, with income)."""
    return df[
        (df["AGEP"] > 16) & (df["PINCP"] > 100) & (df["WKHP"] > 0) & (df["PWGTP"] >= 1)
    ]


def build_credit() -> pd.DataFrame:
    """Download ACS, decode to readable features, and sample a fixed panel."""
    src = ACSDataSource(
        survey_year=YEAR,
        horizon="1-Year",
        survey="person",
        root_dir=str(PROCESSED.parent / "external" / "acs"),
    )
    print(f"Downloading ACS {YEAR} 1-year person data for {STATES} ...")
    raw = src.get_data(states=STATES, download=True)
    df = _acsincome_filter(raw)
    print(f"  {len(raw):,} raw rows -> {len(df):,} after the ACSIncome filter")

    out = pd.DataFrame(
        {
            "age": df["AGEP"].astype(int),
            "education_years": df["SCHL"].map(_schl_years).astype(int),
            "hours_per_week": df["WKHP"].astype(int),
            "occupation": df["OCCP"].map(_occupation_group),
            "marital": df["MAR"].map(_MARITAL),
            "sex": (df["SEX"] == 1).astype(int),  # 1 = male
            "income_high": (df["PINCP"] > 50_000).astype(int),
        }
    )
    out = out.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)
    gm = out.loc[out.sex == 1, "income_high"].mean()
    gf = out.loc[out.sex == 0, "income_high"].mean()
    print(
        f"  sampled {len(out):,} | {out.sex.mean():.0%} male | "
        f"{out.income_high.mean():.0%} high-income | gap {gm - gf:+.0%}"
    )
    return out


def main() -> None:
    """Build the credit/fairness panel and write it to ``data/processed``."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = build_credit()
    path = PROCESSED / "credit_acs.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
