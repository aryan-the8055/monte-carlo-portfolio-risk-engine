"""Backtests whether each model's predicted VaR actually holds up against real
history (Kupiec exception test), and checks how simulation estimates converge
as path count grows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config, engine, generators, metrics, returns as returns_mod


MIN_INDEPENDENT_WINDOWS = 5  # below this, a chi-square Kupiec p-value isn't meaningful


def realized_portfolio_returns(
    hist_returns: pd.DataFrame, weights: dict[str, float], horizon_days: int, overlapping: bool = True
) -> np.ndarray:
    """Realized simple returns the portfolio would actually have earned
    historically over the given horizon.

    overlapping=True: one window per possible start date (large sample, but
      consecutive windows share almost all their days, so they are *not*
      independent observations — useful as a descriptive statistic, not as
      input to a hypothesis test).
    overlapping=False: non-overlapping back-to-back blocks only (small sample,
      but each observation is genuinely independent — this is what a
      significance test is allowed to assume).
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    port_log_ret = hist_returns[tickers].values @ w

    n_obs = len(port_log_ret)
    if n_obs <= horizon_days:
        raise ValueError("Not enough historical data for this horizon.")

    cum_log = np.concatenate([[0.0], np.cumsum(port_log_ret)])

    if overlapping:
        window_log_sum = cum_log[horizon_days:] - cum_log[:-horizon_days]
    else:
        n_blocks = n_obs // horizon_days
        block_ends = np.arange(1, n_blocks + 1) * horizon_days
        block_starts = block_ends - horizon_days
        window_log_sum = cum_log[block_ends] - cum_log[block_starts]

    return np.exp(window_log_sum) - 1


def kupiec_test(predicted_var: float, realized_returns: np.ndarray, confidence: float) -> dict:
    """Kupiec (1995) proportion-of-failures test: compares the observed VaR
    exception rate to the theoretical one and returns a likelihood-ratio
    statistic + p-value (chi-square, 1 df). A low p-value (< 0.05) means the
    model's exception rate is statistically wrong, not just off by chance.

    Only valid when `realized_returns` are independent observations (i.e.
    non-overlapping windows) — the LR statistic assumes i.i.d. Bernoulli
    exceptions, which overlapping windows violate. If there are too few
    independent windows to say anything (< MIN_INDEPENDENT_WINDOWS), the test
    is explicitly marked inconclusive instead of returning a misleadingly
    confident p-value.
    """
    n = len(realized_returns)
    exceptions = int((realized_returns < -predicted_var).sum())
    p_expected = 1 - confidence
    p_observed = exceptions / n if n else float("nan")

    if n < MIN_INDEPENDENT_WINDOWS:
        return {
            "n_windows": n,
            "exceptions": exceptions,
            "expected_rate": p_expected,
            "observed_rate": p_observed,
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "model_rejected_at_5pct": None,
            "note": f"inconclusive: only {n} independent windows (<{MIN_INDEPENDENT_WINDOWS})",
        }

    eps = 1e-12
    p_o = min(max(p_observed, eps), 1 - eps)

    log_likelihood_null = (n - exceptions) * np.log(1 - p_expected) + exceptions * np.log(p_expected)
    log_likelihood_alt = (n - exceptions) * np.log(1 - p_o) + exceptions * np.log(p_o)
    lr_stat = -2 * (log_likelihood_null - log_likelihood_alt)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "n_windows": n,
        "exceptions": exceptions,
        "expected_rate": p_expected,
        "observed_rate": p_observed,
        "lr_stat": float(lr_stat),
        "p_value": float(p_value),
        "model_rejected_at_5pct": bool(p_value < 0.05),
        "note": "",
    }


def validate_all_models(
    all_runs: dict[tuple[str, int], np.ndarray],
    hist_returns: pd.DataFrame,
    weights: dict[str, float],
    initial_value: float = 100.0,
    var_confidences: list[float] = config.VAR_CONFIDENCES,
) -> pd.DataFrame:
    """Runs the Kupiec test for every (model, horizon, confidence) combo.

    The statistical test (p-value, rejection flag) uses non-overlapping
    windows only, so it's a valid hypothesis test. The overlapping-window
    exception rate is also reported alongside it as a larger-sample
    descriptive statistic — useful for eyeballing the trend, but explicitly
    NOT the basis for the p-value, since overlap makes that statistic
    anti-conservative (overstates significance).
    """
    rows = []
    horizons_seen = sorted({h for _, h in all_runs})
    realized_indep = {
        h: realized_portfolio_returns(hist_returns, weights, h * config.TRADING_DAYS_PER_YEAR, overlapping=False)
        for h in horizons_seen
    }
    realized_overlap = {
        h: realized_portfolio_returns(hist_returns, weights, h * config.TRADING_DAYS_PER_YEAR, overlapping=True)
        for h in horizons_seen
    }

    for (model, horizon), wealth_paths in all_runs.items():
        sim_returns = metrics.terminal_returns(wealth_paths, initial_value)
        for c in var_confidences:
            predicted_var, _ = metrics.var_cvar(sim_returns, c)
            test = kupiec_test(predicted_var, realized_indep[horizon], c)

            overlap_returns = realized_overlap[horizon]
            overlap_rate = float((overlap_returns < -predicted_var).mean())

            rows.append({
                "model": model, "horizon_years": horizon, "confidence": c,
                "predicted_VaR": predicted_var,
                **test,
                "overlapping_observed_rate_descriptive": overlap_rate,
                "n_windows_overlapping_descriptive": len(overlap_returns),
            })

    return pd.DataFrame(rows).sort_values(["horizon_years", "confidence", "model"]).reset_index(drop=True)


def long_history_stress_test(
    spx_daily_log_returns: pd.Series,
    shiller_monthly_total_return: pd.Series,
    horizon_years: int = 10,
    n_paths: int = config.N_PATHS,
    block_size_months: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Single-asset (equity-only) block bootstrap comparing two source pools:
    'ETF-era only' (yfinance, ~1993+) vs 'full history' (Shiller, 1871+, which
    includes the 1929 crash, 1937, 1973-74, 1987, 2000-02 and 2008). Shows
    numerically how much tail risk a modern-data-only bootstrap misses because
    it has never seen a Depression-era or 1970s-stagflation regime.
    """
    etf_monthly_log = returns_mod.monthly_log_returns_from_daily(spx_daily_log_returns)
    etf_monthly_simple = np.exp(etf_monthly_log) - 1
    horizon_months = horizon_years * 12

    def bootstrap_terminal(simple_return_series: pd.Series) -> np.ndarray:
        log_ret = pd.DataFrame({"eq": np.log1p(simple_return_series.values)})
        paths = generators.generate_block_bootstrap(
            log_ret, n_paths, horizon_months, block_size=block_size_months, seed=seed
        )
        return engine.simulate_portfolio(paths, weights=np.array([1.0]))

    wealth_etf_era = bootstrap_terminal(etf_monthly_simple)
    wealth_full_history = bootstrap_terminal(shiller_monthly_total_return)

    row_etf = {"source": "ETF-era only (yfinance, ~1993+)", "n_months": len(etf_monthly_simple)}
    row_etf.update(metrics.summarize(wealth_etf_era))
    row_full = {"source": "Full history (Shiller, 1871+)", "n_months": len(shiller_monthly_total_return)}
    row_full.update(metrics.summarize(wealth_full_history))

    return pd.DataFrame([row_etf, row_full])


def convergence_analysis(
    mu: pd.Series,
    cov: pd.DataFrame,
    hist_returns: pd.DataFrame,
    weights: dict[str, float],
    horizon_years: int,
    generator_name: str = "gaussian",
    path_counts: list[int] = config.CONVERGENCE_PATH_COUNTS,
    seed: int = 42,
) -> pd.DataFrame:
    """Re-runs one generator at increasing path counts and tracks how VaR/CVaR
    estimates stabilize, to justify the chosen n_paths rather than picking it
    arbitrarily.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    horizon_days = horizon_years * config.TRADING_DAYS_PER_YEAR

    rows = []
    for n_paths in path_counts:
        if generator_name == "gaussian":
            paths = generators.generate_gaussian(mu[tickers], cov.loc[tickers, tickers], n_paths, horizon_days, seed=seed)
        elif generator_name == "student_t":
            paths = generators.generate_student_t(mu[tickers], cov.loc[tickers, tickers], n_paths, horizon_days, seed=seed)
        elif generator_name == "block_bootstrap":
            paths = generators.generate_block_bootstrap(hist_returns[tickers], n_paths, horizon_days, seed=seed)
        elif generator_name == "garch":
            paths = generators.generate_garch(mu[tickers], hist_returns[tickers], n_paths, horizon_days, seed=seed)
        else:
            raise ValueError(f"Unknown generator: {generator_name}")

        wealth = engine.simulate_portfolio(paths, w)
        row = {"n_paths": n_paths}
        row.update(metrics.summarize(wealth))
        rows.append(row)

    return pd.DataFrame(rows)
