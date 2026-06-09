"""Download and prepare the real market data the course runs on.

Source: daily prices from **Yahoo Finance** via ``yfinance``. We pull a curated
multi-sector *basket* of large-cap names, take Yahoo's split/dividend-**adjusted**
close, and derive daily log-returns. The panel is small enough to ship inside the
repo and to plot legibly.

The basket is chosen so the dependence lessons are visible to the naked eye:
banks that crash together, a couple of mega-cap tech names, energy, and some
defensive utilities/staples that march to a different drum.

The window spans several nameable shocks the course leans on: the 2015 China
"Black Monday", the 2016 Brexit vote, the Feb-2018 "Volmageddon", the Mar-2020
COVID crash, the 2022 drawdown, and the Mar-2023 SVB banking scare.

Run:  python scripts/download_market_data.py
      (requires ``yfinance``; install with ``uv run --with yfinance`` or add the
      dependency to your environment)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# Inclusive start, exclusive end (Yahoo convention). The window spans both the
# 2008 global financial crisis and the 2020 COVID crash — the two canonical
# bank co-crashes the course leans on.
START = "2007-01-01"
END = "2025-12-31"

# Curated basket: ticker -> (human label, sector). Sector lets us colour the
# dependence map and tell a story about *why* some pairs co-move. We use GOOG
# (Class C) rather than GOOGL: Yahoo's GOOGL history only begins at the Apr-2014
# share-class split, while GOOG runs continuously from 2004.
BASKET: dict[str, tuple[str, str]] = {
    "JPM": ("JPMorgan", "Banks"),
    "BAC": ("Bank of Am.", "Banks"),
    "C": ("Citigroup", "Banks"),
    "WFC": ("Wells Fargo", "Banks"),
    "GS": ("Goldman", "Banks"),
    "AAPL": ("Apple", "Tech"),
    "MSFT": ("Microsoft", "Tech"),
    "GOOG": ("Alphabet", "Tech"),
    "XOM": ("Exxon", "Energy"),
    "CVX": ("Chevron", "Energy"),
    "DUK": ("Duke Energy", "Utilities"),
    "SO": ("Southern Co.", "Utilities"),
    "PG": ("P&G", "Staples"),
    "KO": ("Coca-Cola", "Staples"),
}

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def _download_prices() -> pd.DataFrame:
    """Fetch split/dividend-adjusted daily closes for the basket from Yahoo."""
    tickers = list(BASKET)
    print(f"Downloading {len(tickers)} tickers from Yahoo ({START} -> {END}) ...")
    raw = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=True,  # 'Close' becomes the split/dividend-adjusted price
        progress=False,
        threads=False,  # serial is slower but far more robust to rate limits
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo returned no data (rate-limited or offline?)")
    # Multi-ticker downloads come back with a (field, ticker) column MultiIndex.
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.reindex(columns=tickers)  # stable, basket-ordered columns
    close.index.name = "date"
    returned = [t for t in tickers if not close[t].isna().all()]
    print(
        f"  {close.shape[0]:,} trading days, "
        f"{len(returned)} of {len(tickers)} tickers returned, "
        f"{close.index.min().date()} -> {close.index.max().date()}"
    )
    return close


def build_returns() -> pd.DataFrame:
    """Build a wide daily log-return panel for the curated basket."""
    prices = _download_prices().dropna(how="any")
    # daily log-returns: r_t = log(P_t / P_{t-1})
    rets = np.log(prices / prices.shift(1)).dropna(how="any")
    print(f"  basket returns: {rets.shape[0]:,} days x {rets.shape[1]} assets")
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
