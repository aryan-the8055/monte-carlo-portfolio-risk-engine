"""Risk metrics computed from simulated wealth paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def terminal_returns(wealth_paths: np.ndarray, initial_value: float = 100.0) -> np.ndarray:
    return wealth_paths[:, -1] / initial_value - 1


def var_cvar(simple_returns: np.ndarray, confidence: float) -> tuple[float, float]:
    """VaR/CVaR as positive loss fractions at the given confidence level
    (e.g. confidence=0.95 -> the loss exceeded only 5% of the time).
    """
    var = -np.percentile(simple_returns, (1 - confidence) * 100)
    tail = simple_returns[simple_returns <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return float(var), float(cvar)


def max_drawdown_per_path(wealth_paths: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (wealth_paths - running_max) / running_max
    return drawdowns.min(axis=1)  # most negative drawdown per path, e.g. -0.35


def prob_large_drawdown(wealth_paths: np.ndarray, threshold: float = config.DRAWDOWN_THRESHOLD) -> float:
    mdd = max_drawdown_per_path(wealth_paths)
    return float((mdd <= -threshold).mean())


def prob_loss(wealth_paths: np.ndarray, initial_value: float = 100.0) -> float:
    return float((wealth_paths[:, -1] < initial_value).mean())


def terminal_wealth_stats(wealth_paths: np.ndarray) -> dict[str, float]:
    terminal = wealth_paths[:, -1]
    return {
        "mean": float(terminal.mean()),
        "median": float(np.median(terminal)),
        "p5": float(np.percentile(terminal, 5)),
        "p95": float(np.percentile(terminal, 95)),
    }


def summarize(
    wealth_paths: np.ndarray,
    initial_value: float = 100.0,
    var_confidences: list[float] = config.VAR_CONFIDENCES,
    drawdown_threshold: float = config.DRAWDOWN_THRESHOLD,
) -> dict[str, float]:
    """One flat dict of headline risk metrics for a single model/horizon run."""
    ret = terminal_returns(wealth_paths, initial_value)
    row: dict[str, float] = {}

    for c in var_confidences:
        var, cvar = var_cvar(ret, c)
        row[f"VaR_{int(c * 100)}"] = var
        row[f"CVaR_{int(c * 100)}"] = cvar

    row[f"P_drawdown_gt_{int(drawdown_threshold * 100)}pct"] = prob_large_drawdown(
        wealth_paths, drawdown_threshold
    )
    row["P_loss"] = prob_loss(wealth_paths, initial_value)

    for k, v in terminal_wealth_stats(wealth_paths).items():
        row[f"terminal_wealth_{k}"] = v

    return row


def build_results_table(all_runs: dict[tuple[str, int], np.ndarray], initial_value: float = 100.0) -> pd.DataFrame:
    """all_runs: {(model_name, horizon_years): wealth_paths} -> one tidy row per
    (model, horizon), fixed schema, ready to write to results/results.csv.
    """
    rows = []
    for (model, horizon), wealth_paths in all_runs.items():
        row = {"model": model, "horizon_years": horizon}
        row.update(summarize(wealth_paths, initial_value=initial_value))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["horizon_years", "model"]).reset_index(drop=True)
