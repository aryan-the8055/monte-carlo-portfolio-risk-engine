"""Three return-path generators sharing one interface:

    generate_X(..., n_paths, horizon_days) -> np.ndarray of shape
        (n_paths, horizon_days, n_assets)

containing simulated *daily log returns* per asset. The engine turns these
into wealth paths; nothing here knows about weights or dollars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def generate_gaussian(
    mu: pd.Series, cov: pd.DataFrame, n_paths: int, horizon_days: int, seed: int | None = None
) -> np.ndarray:
    """Multivariate normal daily returns from annualized mu/cov."""
    rng = np.random.default_rng(seed)
    daily_mu = mu.values / config.TRADING_DAYS_PER_YEAR
    daily_cov = cov.values / config.TRADING_DAYS_PER_YEAR
    return rng.multivariate_normal(daily_mu, daily_cov, size=(n_paths, horizon_days))


def generate_student_t(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_paths: int,
    horizon_days: int,
    dof: int = config.STUDENT_T_DOF,
    seed: int | None = None,
) -> np.ndarray:
    """Multivariate Student-t daily returns (fat tails), same target mean/cov as
    the Gaussian generator so the two are directly comparable — only the tail
    shape differs.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    daily_mu = mu.values / config.TRADING_DAYS_PER_YEAR
    daily_cov = cov.values / config.TRADING_DAYS_PER_YEAR

    chol = np.linalg.cholesky(daily_cov)
    z = rng.standard_normal((n_paths, horizon_days, n_assets))
    chi2 = rng.chisquare(dof, size=(n_paths, horizon_days, 1))
    t_std = z / np.sqrt(chi2 / dof)
    t_std *= np.sqrt((dof - 2) / dof)  # rescale so Var matches daily_cov (dof > 2)

    return daily_mu + t_std @ chol.T


def generate_block_bootstrap(
    hist_returns: pd.DataFrame,
    n_paths: int,
    horizon_days: int,
    block_size: int = config.BLOCK_SIZE_DAYS,
    seed: int | None = None,
) -> np.ndarray:
    """Resample contiguous blocks of real historical multi-asset daily returns.
    Sampling whole rows (all assets together) for each day preserves the
    real cross-asset correlation and each block's real autocorrelation /
    volatility clustering, unlike independent per-asset resampling.
    """
    rng = np.random.default_rng(seed)
    hist = hist_returns.values
    n_obs, n_assets = hist.shape

    n_blocks = int(np.ceil(horizon_days / block_size))
    max_start = n_obs - block_size
    if max_start < 1:
        raise ValueError("Not enough historical observations for the requested block size.")

    starts = rng.integers(0, max_start, size=(n_paths, n_blocks))
    offsets = np.arange(block_size)
    idx = starts[:, :, None] + offsets[None, None, :]          # (n_paths, n_blocks, block_size)
    idx = idx.reshape(n_paths, n_blocks * block_size)

    paths = hist[idx]  # fancy-index -> (n_paths, n_blocks*block_size, n_assets)
    return paths[:, :horizon_days, :]


def generate_garch(
    mu: pd.Series,
    daily_log_returns: pd.DataFrame,
    n_paths: int,
    horizon_days: int,
    seed: int | None = None,
) -> np.ndarray:
    """Constant Conditional Correlation GARCH(1,1) (Bollerslev, 1990): fit a
    univariate GARCH(1,1) to each asset separately (so each asset gets its own
    volatility-clustering dynamics — calm periods stay calm, crashes stay
    volatile), then induce cross-asset correlation on the daily shocks using
    the historical correlation matrix, held constant over the horizon.

    This is the one generator here that captures volatility clustering; the
    other three treat each day as independently drawn.

    `mu` (annualized) is the SAME externally-supplied target mean used by
    `generate_gaussian`/`generate_student_t`, applied here too — each asset's
    series is demeaned to that target *before* fitting, and the model is fit
    with a zero mean. This matters: letting GARCH estimate its own
    mean-likelihood-weighted mu (the `arch` default) pulls the drift higher
    than the plain historical average, because the GARCH likelihood
    down-weights large negative returns that occur in high-volatility
    regimes — which would make this generator's growth not comparable to the
    other three. Holding mu fixed isolates the comparison to "does modeling
    volatility clustering change the risk estimate," which is the point.
    """
    from arch import arch_model  # local import: optional, heavier dependency

    rng = np.random.default_rng(seed)
    tickers = list(daily_log_returns.columns)
    n_assets = len(tickers)
    daily_mu = mu.values / config.TRADING_DAYS_PER_YEAR

    omega = np.zeros(n_assets)
    alpha = np.zeros(n_assets)
    beta = np.zeros(n_assets)
    last_sigma2 = np.zeros(n_assets)
    last_resid = np.zeros(n_assets)

    for i, ticker in enumerate(tickers):
        demeaned_pct = (daily_log_returns[ticker].values - daily_mu[i]) * 100  # arch needs a non-tiny scale
        model = arch_model(demeaned_pct, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")
        params = fit.params
        omega[i] = params["omega"] / 100**2
        alpha[i] = params["alpha[1]"]
        beta[i] = params["beta[1]"]
        last_sigma2[i] = (fit.conditional_volatility[-1] / 100) ** 2
        last_resid[i] = fit.resid[-1] / 100

    corr = daily_log_returns.corr().values
    chol = np.linalg.cholesky(corr)

    sigma2 = np.tile(last_sigma2, (n_paths, 1))  # (n_paths, n_assets), GARCH state
    resid = np.tile(last_resid, (n_paths, 1))

    paths = np.empty((n_paths, horizon_days, n_assets))
    for t in range(horizon_days):
        sigma2 = omega + alpha * resid**2 + beta * sigma2
        sigma = np.sqrt(sigma2)
        z = rng.standard_normal((n_paths, n_assets)) @ chol.T  # unit-variance, correlated
        resid = sigma * z
        paths[:, t, :] = daily_mu + resid

    return paths


GENERATORS = {
    "gaussian": generate_gaussian,
    "student_t": generate_student_t,
    "block_bootstrap": generate_block_bootstrap,
    "garch": generate_garch,
}
