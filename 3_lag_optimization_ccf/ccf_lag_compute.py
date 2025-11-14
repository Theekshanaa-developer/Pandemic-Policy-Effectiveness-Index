"""
3_lag_optimization_ccf/ccf_lag_compute.py
Computes stringency → cases lag using Cross-Correlation Function (CCF)
for each country and saves results + plots.

Outputs:
- lag_results.csv
- lag_distribution.png
- india_ccf_plot.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

from utils.preprocess import load_data
from utils.common import ensure_outdir

OUT_DIR = os.path.join("3_lag_optimization_ccf", "outputs")
ensure_outdir(OUT_DIR)


def compute_ccf_lag(series_x, series_y, max_lag=30):
    """
    Compute lag between stringency (x) and cases (y).
    Returns lag in days where x leads y.
    """
    x = (series_x - series_x.mean()) / series_x.std()
    y = (series_y - series_y.mean()) / series_y.std()

    corr = correlate(y, x, mode='full')   # x leads y
    lags = np.arange(-len(y) + 1, len(x))

    # Restrict lags to ± max_lag
    mask = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr[mask]
    lags = lags[mask]

    lag = lags[np.argmax(corr)]
    return lag


def compute_lags_for_all(df):
    """Compute lags for each country."""
    results = []

    for country, group in df.groupby("country"):
        if group["stringency_index"].nunique() < 5:
            continue

        lag = compute_ccf_lag(group["stringency_index"], group["cases_7d"])
        results.append([country, lag])

    return pd.DataFrame(results, columns=["country", "lag_days"])


def plot_lag_distribution(df, outpath):
    """Plot histogram of lags."""
    plt.figure(figsize=(8, 5))
    plt.hist(df["lag_days"], bins=15, edgecolor="black")
    plt.title("Lag Distribution Across Countries")
    plt.xlabel("Lag (days)")
    plt.ylabel("Count")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_india_ccf(india_df, outpath):
    """Plot India’s CCF for interpretation."""
    x = india_df["stringency_index"]
    y = india_df["cases_7d"]

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    corr = correlate(y, x, mode='full')
    lags = np.arange(-len(y) + 1, len(x))

    # Restrict to meaningful range
    mask = (lags >= -30) & (lags <= 30)
    corr = corr[mask]
    lags = lags[mask]

    plt.figure(figsize=(9, 5))
    plt.plot(lags, corr)
    plt.title("India – Cross Correlation (Stringency → Cases)")
    plt.xlabel("Lag (days)")
    plt.ylabel("Correlation Strength")
    plt.axvline(x=lags[np.argmax(corr)], color="red", linestyle="--")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    df = load_data()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"])

    # Compute for all countries
    lag_df = compute_lags_for_all(df)
    lag_df.to_csv(os.path.join(OUT_DIR, "lag_results.csv"), index=False)

    # Lag distribution
    plot_lag_distribution(lag_df, os.path.join(OUT_DIR, "lag_distribution.png"))

    # India only
    india = df[df["country"].str.lower() == "india"].sort_values("date")
    plot_india_ccf(india, os.path.join(OUT_DIR, "india_ccf_plot.png"))

    print("Lag computation complete.")
