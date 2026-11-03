"""Plots: fan chart, VaR/CVaR comparison, convergence. Each function saves a
PNG to results/ and returns the matplotlib Figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

PERCENTILE_BANDS = [(5, 95), (25, 75)]


def fan_chart(wealth_paths: np.ndarray, title: str, filename: str) -> plt.Figure:
    n_paths, n_steps = wealth_paths.shape
    x = np.arange(n_steps)
    median = np.median(wealth_paths, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    for lo, hi in PERCENTILE_BANDS:
        lo_band = np.percentile(wealth_paths, lo, axis=0)
        hi_band = np.percentile(wealth_paths, hi, axis=0)
        ax.fill_between(x, lo_band, hi_band, alpha=0.25, label=f"{lo}-{hi}th pct")
    ax.plot(x, median, color="black", linewidth=1.5, label="median")
    ax.set_title(title)
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Portfolio value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / filename, dpi=150)
    return fig


def var_comparison_bar(results_df: pd.DataFrame, horizon_years: int, filename: str) -> plt.Figure:
    sub = results_df[results_df["horizon_years"] == horizon_years]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sub))
    width = 0.35
    ax.bar(x - width / 2, sub["VaR_95"], width, label="VaR 95%")
    ax.bar(x + width / 2, sub["CVaR_95"], width, label="CVaR 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model"])
    ax.set_ylabel("Loss fraction")
    ax.set_title(f"VaR vs CVaR by model — {horizon_years}y horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / filename, dpi=150)
    return fig


def convergence_plot(convergence_df: pd.DataFrame, metric: str, filename: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(convergence_df["n_paths"], convergence_df[metric], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel(metric)
    ax.set_title(f"Convergence of {metric} vs. simulation count")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / filename, dpi=150)
    return fig
