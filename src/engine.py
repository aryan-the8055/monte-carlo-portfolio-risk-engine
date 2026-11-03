"""Turns simulated asset-return paths into portfolio wealth paths, and runs the
full model x horizon grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, generators


def simulate_portfolio(
    asset_return_paths: np.ndarray,
    weights: np.ndarray,
    initial_value: float = 100.0,
    contribution: float = 0.0,
    contribution_freq_days: int | None = None,
) -> np.ndarray:
    """Daily-rebalanced portfolio wealth paths.

    asset_return_paths: (n_paths, horizon_days, n_assets) daily log returns
    weights: (n_assets,) target weights, held constant via daily rebalancing
             (so portfolio log return each day = weighted sum of asset log
             returns — the standard first-order approximation for frequent
             rebalancing).
    Returns: (n_paths, horizon_days + 1) wealth paths, including t=0.
    """
    w = np.asarray(weights)
    portfolio_log_returns = asset_return_paths @ w  # (n_paths, horizon_days)
    simple_returns = np.exp(portfolio_log_returns) - 1
    n_paths, horizon_days = simple_returns.shape

    if contribution == 0 or contribution_freq_days is None:
        growth = np.cumprod(1 + simple_returns, axis=1)
        wealth = np.concatenate(
            [np.full((n_paths, 1), initial_value), initial_value * growth], axis=1
        )
        return wealth

    wealth = np.empty((n_paths, horizon_days + 1))
    wealth[:, 0] = initial_value
    for t in range(horizon_days):
        wealth[:, t + 1] = wealth[:, t] * (1 + simple_returns[:, t])
        if (t + 1) % contribution_freq_days == 0:
            wealth[:, t + 1] += contribution
    return wealth


def run_all_models(
    mu: pd.Series,
    cov: pd.DataFrame,
    hist_returns: pd.DataFrame,
    weights: dict[str, float],
    horizons_years: list[int] = config.HORIZONS_YEARS,
    n_paths: int = config.N_PATHS,
    seed: int | None = 42,
) -> dict[tuple[str, int], np.ndarray]:
    """Runs all three generators across all horizons for the given portfolio.
    Returns {(model_name, horizon_years): wealth_paths} so metrics/validation
    can iterate over the full grid.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    mu_sub, cov_sub = mu[tickers], cov.loc[tickers, tickers]
    hist_sub = hist_returns[tickers]

    out: dict[tuple[str, int], np.ndarray] = {}
    for horizon_years in horizons_years:
        horizon_days = horizon_years * config.TRADING_DAYS_PER_YEAR

        gauss_paths = generators.generate_gaussian(mu_sub, cov_sub, n_paths, horizon_days, seed=seed)
        out[("gaussian", horizon_years)] = simulate_portfolio(gauss_paths, w)

        t_paths = generators.generate_student_t(mu_sub, cov_sub, n_paths, horizon_days, seed=seed)
        out[("student_t", horizon_years)] = simulate_portfolio(t_paths, w)

        boot_paths = generators.generate_block_bootstrap(hist_sub, n_paths, horizon_days, seed=seed)
        out[("block_bootstrap", horizon_years)] = simulate_portfolio(boot_paths, w)

        garch_paths = generators.generate_garch(mu_sub, hist_sub, n_paths, horizon_days, seed=seed)
        out[("garch", horizon_years)] = simulate_portfolio(garch_paths, w)

    return out
