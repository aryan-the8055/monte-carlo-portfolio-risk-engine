# Monte Carlo Portfolio Risk Engine

VaR, CVaR and drawdown-probability simulator for a multi-asset portfolio,
built to compare four return-generating models head-to-head and prove —
empirically, via backtesting — which assumptions actually hold up against
real history.

See [CLAUDE.md](CLAUDE.md) for the full problem definition, dataset list and
build plan.

## What it does

1. Downloads/caches multi-asset price history (`yfinance`), long-history S&P
   500 data (Shiller, 1871+), and macro series (FRED CPI, T-bill).
2. Simulates thousands of portfolio paths under four models, all sharing the
   same target mean/covariance so only the tail/dynamics assumption differs:
   Gaussian, Student-t (fat tails), block bootstrap (resampled real history),
   and GARCH(1,1) with constant conditional correlation (volatility
   clustering — the only one of the four where a calm period stays calm and
   a volatile one stays volatile, instead of every day being drawn
   independently).
3. Computes VaR/CVaR (95%/99%), probability of a >20% drawdown, and terminal
   wealth distribution for each model x horizon (1y/5y/10y).
4. **Validates** each model with a Kupiec exception test against real
   historical **non-overlapping** windows — the statistically valid sample
   for a hypothesis test — and reports the larger overlapping-window
   exception rate alongside it as a labeled descriptive statistic only, since
   overlap makes that version anti-conservative.
5. Runs a long-history stress test comparing an "ETF-era only" bootstrap
   (post-1993) against a "full history" bootstrap (Shiller, since 1871,
   includes 1929/1937/1973-74/1987/2000/2008) to quantify how much tail risk
   modern-only data misses.
6. Produces fan charts, a VaR/CVaR comparison chart, and a convergence plot.

## Setup

```bash
pip install -r requirements.txt
python run.py
```

Outputs land in `results/`:
- `model_comparison.xlsx` — one workbook, one sheet per table: **Overview**
  (run config), **Results** (VaR/CVaR/drawdown/terminal wealth per model x
  horizon), **Kupiec Validation**, **Convergence**, **Long-History Stress
  Test**.
- 5 PNG charts (fan chart per model, VaR/CVaR comparison, convergence plot).

Raw/cached data lives in `data/raw/` and `data/processed/`, with source
documentation auto-written to `data/README.md` on first run.

## Default portfolio (editable in `src/config.py`)

| Ticker | Weight | Role |
|---|---|---|
| SPY | 45% | US equities |
| QQQ | 15% | Tech-tilted equities |
| AGG | 30% | Bonds |
| GLD | 10% | Gold |

## Data sources & licensing

- **yfinance** — Yahoo Finance adjusted daily prices, personal/research use
  only, pulled with `auto_adjust=True` (total-return series).
- **Shiller S&P 500 dataset** (datahub.io mirror) — public, monthly, 1871+.
- **FRED** (`CPIAUCSL`, `TB3MS`) — public domain, no API key required.

Full coverage dates and caveats are written automatically to
`data/README.md` after the first data pull (ticker inception dates vary, so
the common estimation window is capped by the youngest ticker).

## Methodology notes / guardrails

- Daily-rebalanced portfolio approximation: each day's portfolio return is
  the weight-average of asset log returns (standard first-order
  approximation for frequent rebalancing).
- VaR/CVaR are computed on simulated **terminal** horizon returns, not daily
  returns.
- GARCH is fit on each asset's returns **demeaned to the same target mean**
  used by the Gaussian/Student-t generators, with a zero-mean GARCH(1,1) on
  the residual. Letting `arch`'s own MLE estimate the mean instead would pull
  the drift above the plain historical average (the GARCH likelihood
  down-weights large negative returns, which occur in high-volatility
  regimes) and make its results not comparable to the other three models.
- The Kupiec test's p-value/rejection flag use **non-overlapping** historical
  windows only — each one an independent observation, which the underlying
  chi-square test requires. With ~22 years of daily data, that leaves 21
  independent 1-year windows, 4 for 5-year, and only 2 for 10-year — the
  5y/10y tests are correctly reported as **inconclusive** rather than
  returning a misleadingly confident p-value from too few samples. The larger
  overlapping-window exception rate is also reported, labeled explicitly as
  descriptive-only (not a hypothesis-test input), since overlap inflates the
  effective sample size and overstates significance.
- Results are evidence from historical/simulated data, not a guarantee of
  future performance.
- AI-assisted work: project scaffolding, module structure, and boilerplate
  code were drafted with Claude (Anthropic); methodology, parameter choices
  and validation design were directed and reviewed by the author.
