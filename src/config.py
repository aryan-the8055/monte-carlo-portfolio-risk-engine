"""Central constants: tickers, paths, dates. Nothing else imports network/IO libs from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

for _d in (DATA_RAW, DATA_PROCESSED, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Multi-asset basket: equities, bonds, gold, real estate, cash-like
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "AGG", "IEF", "TLT", "GLD", "VNQ", "BIL"]

# Default portfolio used for the MVP run (equal-weight across a simpler 4-asset
# subset so the headline results are easy to reason about; full basket is still
# downloaded/cached for extensions).
DEFAULT_PORTFOLIO = {
    "SPY": 0.45,   # US equities
    "QQQ": 0.15,   # tech-tilted equities
    "AGG": 0.30,   # bonds
    "GLD": 0.10,   # gold
}

YF_START = "1993-01-01"  # SPY inception; common window gets trimmed per-ticker anyway

SHILLER_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
SHILLER_CSV = DATA_RAW / "shiller_sp500.csv"

FRED_SERIES = {
    "CPIAUCSL": "CPI, all urban consumers (monthly) — inflation",
    "TB3MS": "3-month T-bill secondary market rate (monthly) — risk-free rate",
}
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

ETF_PRICES_CSV = DATA_RAW / "etf_prices.csv"
RETURNS_CSV = DATA_PROCESSED / "log_returns.csv"

TRADING_DAYS_PER_YEAR = 252

# Simulation defaults
N_PATHS = 10_000
HORIZONS_YEARS = [1, 5, 10]
CONVERGENCE_PATH_COUNTS = [1_000, 5_000, 10_000, 50_000]
STUDENT_T_DOF = 5          # fat-tail parameter; lower = fatter tails
BLOCK_SIZE_DAYS = 21       # ~1 trading month, preserves short-run autocorrelation
DRAWDOWN_THRESHOLD = 0.20  # "large drawdown" = -20% from peak
VAR_CONFIDENCES = [0.95, 0.99]
