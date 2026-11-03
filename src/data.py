"""Download + cache all raw data. Nothing here mutates data/raw/ once written —
cleaned/aligned series go to data/processed/ instead.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests
import yfinance as yf

from . import config


def download_etf_prices(force: bool = False) -> pd.DataFrame:
    """Daily total-return-adjusted close prices for the asset basket, cached to CSV."""
    if config.ETF_PRICES_CSV.exists() and not force:
        return pd.read_csv(config.ETF_PRICES_CSV, index_col=0, parse_dates=True)

    raw = yf.download(config.TICKERS, start=config.YF_START, auto_adjust=True)["Close"]
    raw = raw.sort_index()
    raw.to_csv(config.ETF_PRICES_CSV)
    return raw


def download_shiller_data(force: bool = False) -> pd.DataFrame:
    """Monthly S&P 500 + dividends + CPI history back to 1871 (datahub.io mirror
    of Shiller's dataset). Used as the long-history pool for the block bootstrap.
    """
    if config.SHILLER_CSV.exists() and not force:
        return pd.read_csv(config.SHILLER_CSV, parse_dates=["Date"])

    resp = requests.get(config.SHILLER_URL, timeout=30)
    resp.raise_for_status()
    config.SHILLER_CSV.write_bytes(resp.content)
    return pd.read_csv(config.SHILLER_CSV, parse_dates=["Date"])


def download_fred_series(series_id: str, force: bool = False) -> pd.DataFrame:
    """A single FRED series as a DataFrame indexed by date. No API key required —
    uses the public fredgraph.csv endpoint.
    """
    path = config.DATA_RAW / f"fred_{series_id}.csv"
    if path.exists() and not force:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    url = config.FRED_CSV_URL.format(series_id=series_id)
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    df.to_csv(path)
    return df


def download_all(force: bool = False) -> dict[str, pd.DataFrame]:
    etf_prices = download_etf_prices(force=force)
    shiller = download_shiller_data(force=force)
    fred = {sid: download_fred_series(sid, force=force) for sid in config.FRED_SERIES}

    _write_data_readme(etf_prices, shiller, fred)

    return {"etf_prices": etf_prices, "shiller": shiller, **fred}


def _write_data_readme(etf_prices: pd.DataFrame, shiller: pd.DataFrame, fred: dict) -> None:
    lines = [
        "# Data Sources",
        "",
        f"Generated: {dt.date.today().isoformat()}",
        "",
        "## 1. yfinance — daily adjusted prices (primary)",
        f"- Tickers: {', '.join(config.TICKERS)}",
        "- Pulled with `auto_adjust=True` (dividends folded into price -> total return).",
        f"- Requested start: {config.YF_START}",
        "- Actual coverage per ticker (first non-null date):",
    ]
    for t in config.TICKERS:
        if t in etf_prices.columns:
            first_valid = etf_prices[t].first_valid_index()
            lines.append(f"  - {t}: {first_valid.date() if first_valid is not None else 'N/A'}")
    lines += [
        "- License: Yahoo Finance data, personal/research use only (not for redistribution).",
        "",
        "## 2. Shiller S&P 500 dataset (datahub.io mirror)",
        f"- Source: {config.SHILLER_URL}",
        f"- Rows: {len(shiller)}; date range: {shiller['Date'].min().date()} to {shiller['Date'].max().date()}",
        "- Monthly S&P 500 index, dividends, earnings, CPI, long rate.",
        "- NOTE: recent extension rows (last ~couple years) lack dividend data — "
        "the dividend-based total-return series must be cut off where that column "
        "goes null, not silently forward-filled.",
        "",
        "## 3. FRED series",
    ]
    for sid, desc in config.FRED_SERIES.items():
        df = fred[sid]
        lines.append(f"- {sid} ({desc}): {len(df)} obs, {df.index.min().date()} to {df.index.max().date()}")
    lines += [
        "- Public CSV endpoint, no API key required.",
        "- NOTE: used for inflation/risk-free-rate context only, not as a primary "
        "equity return series.",
        "",
        "## 4. Ken French Data Library (not pulled automatically)",
        "- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        "- Optional academic benchmark cross-check; download manually if used.",
    ]
    (config.ROOT / "data" / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    data = download_all()
    for name, df in data.items():
        print(name, df.shape)
