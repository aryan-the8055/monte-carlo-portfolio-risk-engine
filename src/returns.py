"""Return/covariance estimation from cached price data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def compute_log_returns(prices: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    """Daily log returns, aligned on the common window across the given tickers
    (defaults to all columns). Missing leading history (younger tickers) is
    trimmed, not filled.
    """
    cols = tickers or list(prices.columns)
    px = prices[cols].dropna(how="any")
    log_ret = np.log(px / px.shift(1)).dropna()
    log_ret.to_csv(config.RETURNS_CSV)
    return log_ret


def annualized_stats(log_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualized mean return vector and covariance matrix from daily log returns."""
    mu = log_returns.mean() * config.TRADING_DAYS_PER_YEAR
    cov = log_returns.cov() * config.TRADING_DAYS_PER_YEAR
    return mu, cov


def monthly_log_returns_from_daily(daily_log_returns: pd.Series) -> pd.Series:
    """Aggregate daily log returns to monthly by summing within each calendar
    month — used to put ETF-era data on the same monthly grid as the Shiller
    dataset for the long-history stress test.
    """
    return daily_log_returns.groupby(daily_log_returns.index.to_period("M")).sum()


def shiller_long_history_returns(shiller: pd.DataFrame) -> pd.Series:
    """Monthly nominal total-return series (price + dividend) from the Shiller
    dataset, cut off where dividend data goes missing (the dataset's recent
    'extension' rows only populate the price column).
    """
    df = shiller.sort_values("Date").copy()
    df = df.rename(columns={c: c.strip() for c in df.columns})

    price_col = "SP500" if "SP500" in df.columns else "Real Price"
    div_col = "Dividend" if "Dividend" in df.columns else None

    if div_col is not None:
        df = df[df[div_col].notna()]
        monthly_div_yield = (df[div_col] / 12.0) / df[price_col]
    else:
        monthly_div_yield = 0.0

    price_return = df[price_col].pct_change()
    total_return = price_return + monthly_div_yield
    total_return.index = df["Date"]
    return total_return.dropna()


def shiller_real_returns(shiller: pd.DataFrame, nominal_total_return: pd.Series) -> pd.Series:
    """Inflation-adjust the long-history nominal total-return series using the
    Shiller dataset's own CPI column, so the bootstrap pool reflects real
    purchasing-power moves, not nominal ones.
    """
    df = shiller.sort_values("Date").set_index("Date")
    cpi = df["CPI"] if "CPI" in df.columns else df.get("Consumer Price Index")
    cpi_inflation = cpi.pct_change()
    aligned = nominal_total_return.to_frame("nominal").join(
        cpi_inflation.to_frame("cpi_inflation"), how="inner"
    )
    real = (1 + aligned["nominal"]) / (1 + aligned["cpi_inflation"]) - 1
    return real.dropna()
