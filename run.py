"""End-to-end pipeline: download data -> estimate returns -> simulate ->
compute risk metrics -> validate against history -> plot -> write results.

Usage:
    python run.py
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src import config, data, engine, metrics, returns, validation, viz


def main() -> None:
    print("[1/6] Downloading / loading cached data ...")
    raw = data.download_all()
    etf_prices = raw["etf_prices"]
    shiller = raw["shiller"]

    print("[2/6] Computing returns and annualized stats ...")
    tickers = list(config.DEFAULT_PORTFOLIO.keys())
    log_returns = returns.compute_log_returns(etf_prices, tickers=tickers)
    mu, cov = returns.annualized_stats(log_returns)

    print(f"    Common aligned window: {log_returns.index.min().date()} -> {log_returns.index.max().date()}"
          f" ({len(log_returns)} trading days)")

    print("[3/6] Running Monte Carlo simulation grid (gaussian / student_t / block_bootstrap / garch "
          f"x {config.HORIZONS_YEARS} years, {config.N_PATHS:,} paths each) ...")
    all_runs = engine.run_all_models(
        mu=mu, cov=cov, hist_returns=log_returns,
        weights=config.DEFAULT_PORTFOLIO, horizons_years=config.HORIZONS_YEARS,
        n_paths=config.N_PATHS,
    )

    print("[4/6] Computing risk metrics (VaR / CVaR / drawdown / terminal wealth) ...")
    results_df = metrics.build_results_table(all_runs)
    print(results_df.to_string(index=False))

    print("[5/6] Validating models against real history "
          "(Kupiec exception test, non-overlapping windows) ...")
    kupiec_df = validation.validate_all_models(all_runs, log_returns, config.DEFAULT_PORTFOLIO)
    print(kupiec_df.to_string(index=False))

    print("    Running convergence analysis (gaussian, 10y horizon) ...")
    convergence_df = validation.convergence_analysis(
        mu, cov, log_returns, config.DEFAULT_PORTFOLIO, horizon_years=10, generator_name="gaussian"
    )

    print("    Running long-history stress test (ETF-era vs full Shiller history, equity-only) ...")
    spx_daily_log = log_returns["SPY"]
    shiller_monthly_return = returns.shiller_long_history_returns(shiller)
    stress_df = validation.long_history_stress_test(spx_daily_log, shiller_monthly_return, horizon_years=10)
    print(stress_df.to_string(index=False))

    print("[6/6] Generating plots ...")
    key_horizon = config.HORIZONS_YEARS[-1]
    for model in ["gaussian", "student_t", "block_bootstrap", "garch"]:
        viz.fan_chart(
            all_runs[(model, key_horizon)],
            title=f"{model} — {key_horizon}y fan chart",
            filename=f"fan_chart_{model}_{key_horizon}y.png",
        )
    viz.var_comparison_bar(results_df, key_horizon, f"var_comparison_{key_horizon}y.png")
    viz.convergence_plot(convergence_df, "VaR_95", "convergence_VaR_95.png")

    print("    Writing results/model_comparison.xlsx (one workbook, one sheet per table) ...")
    overview_df = pd.DataFrame([
        {"Field": "Generated", "Value": dt.date.today().isoformat()},
        {"Field": "Portfolio", "Value": ", ".join(f"{t} {w:.0%}" for t, w in config.DEFAULT_PORTFOLIO.items())},
        {"Field": "History window", "Value": f"{log_returns.index.min().date()} to {log_returns.index.max().date()}"},
        {"Field": "Trading days in window", "Value": len(log_returns)},
        {"Field": "Simulated paths per model/horizon", "Value": config.N_PATHS},
        {"Field": "Horizons simulated (years)", "Value": ", ".join(str(h) for h in config.HORIZONS_YEARS)},
        {"Field": "Models", "Value": "gaussian, student_t, block_bootstrap, garch"},
        {"Field": "Sheet: Results", "Value": "VaR/CVaR/drawdown/terminal-wealth per model x horizon"},
        {"Field": "Sheet: Kupiec Validation", "Value": "Backtest of predicted VaR vs real history, non-overlapping windows"},
        {"Field": "Sheet: Convergence", "Value": "How Gaussian VaR/CVaR estimates stabilize as path count grows"},
        {"Field": "Sheet: Long-History Stress Test", "Value": "ETF-era (1993+) vs full Shiller history (1871+) equity tail risk"},
    ])

    xlsx_path = config.RESULTS_DIR / "model_comparison.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="Overview", index=False)
        results_df.to_excel(writer, sheet_name="Results", index=False)
        kupiec_df.to_excel(writer, sheet_name="Kupiec Validation", index=False)
        convergence_df.to_excel(writer, sheet_name="Convergence", index=False)
        stress_df.to_excel(writer, sheet_name="Long-History Stress Test", index=False)

    print(f"\nDone. Outputs written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
