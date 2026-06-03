"""Download and prepare the real market data the course runs on.

Source: the S&P 500 "5yr" daily OHLCV dataset hosted in the public
``plotly/datasets`` GitHub repository (~500 names, Feb 2013 - Feb 2018). We pull
it once, derive daily log-returns, and keep a curated multi-sector *basket* small
enough to ship inside the repo and to plot legibly.

The basket is chosen so the dependence lessons are visible to the naked eye:
banks that crash together, a couple of mega-cap tech names, energy, and some
defensive utilities/staples that march to a different drum.

Run:  python scripts/download_market_data.py
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


SP500_5YR_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
)

# Curated basket: ticker -> (human label, sector). Sector lets us colour the
# dependence map and tell a story about *why* some pairs co-move.
BASKET: dict[str, tuple[str, str]] = {
    "JPM": ("JPMorgan", "Banks"),
    "BAC": ("Bank of Am.", "Banks"),
    "C": ("Citigroup", "Banks"),
    "WFC": ("Wells Fargo", "Banks"),
    "GS": ("Goldman", "Banks"),
    "AAPL": ("Apple", "Tech"),
    "MSFT": ("Microsoft", "Tech"),
    "GOOGL": ("Alphabet", "Tech"),
    "XOM": ("Exxon", "Energy"),
    "CVX": ("Chevron", "Energy"),
    "DUK": ("Duke Energy", "Utilities"),
    "SO": ("Southern Co.", "Utilities"),
    "PG": ("P&G", "Staples"),
    "KO": ("Coca-Cola", "Staples"),
}

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def _download_raw() -> pd.DataFrame:
    """Fetch the raw long-format S&P 5yr price table from GitHub."""
    print(f"Downloading S&P 5yr data from {SP500_5YR_URL} ...")
    with urllib.request.urlopen(SP500_5YR_URL) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), parse_dates=["date"])
    print(
        f"  {len(df):,} rows, {df['Name'].nunique()} tickers, "
        f"{df['date'].min().date()} -> {df['date'].max().date()}"
    )
    return df


def build_returns() -> pd.DataFrame:
    """Pivot to a wide daily log-return panel for the curated basket."""
    df = _download_raw()
    wide = df.pivot(index="date", columns="Name", values="close")
    have = [t for t in BASKET if t in wide.columns]
    missing = [t for t in BASKET if t not in wide.columns]
    if missing:
        print(f"  note: not in source, dropping: {missing}")
    prices = wide[have].dropna(how="any")
    # daily log-returns: r_t = log(P_t / P_{t-1})
    rets = np.log(prices / prices.shift(1)).dropna(how="any")
    print(f"  basket returns: {rets.shape[0]} days x {rets.shape[1]} assets")
    return rets


def main() -> None:
    """Build the returns panel + metadata and write both to ``data/processed``."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rets = build_returns()
    out = PROCESSED / "returns_basket.csv"
    rets.to_csv(out)
    # also persist the label/sector metadata so loaders can use it
    meta = pd.DataFrame(
        [(t, BASKET[t][0], BASKET[t][1]) for t in rets.columns],
        columns=["ticker", "label", "sector"],
    )
    meta.to_csv(PROCESSED / "basket_meta.csv", index=False)
    print(f"Saved {out}")
    print(f"Saved {PROCESSED / 'basket_meta.csv'}")


if __name__ == "__main__":
    main()
