"""
utils/feature_engineering.py
Feature creation and per-country metric computations used by PPEI routines.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def compute_first_case_and_first_policy(g: pd.DataFrame,
                                        case_threshold: float = 0.01,
                                        policy_threshold: float = 50.0) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], float]:
    """
    For a single-country dataframe (g) sorted by date, compute:
      - first_case_date (when cases_per_100k >= case_threshold)
      - first_policy_date (when stringency_index >= policy_threshold)
      - lag_days = first_policy - first_case (nan if either missing)
    """
    if g.empty:
        return None, None, np.nan

    # ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(g["date"]):
        g = g.copy()
        g["date"] = pd.to_datetime(g["date"])

    fc = g[g["cases_per_100k"] >= case_threshold]
    first_case = fc["date"].min() if not fc.empty else pd.NaT

    sp = g[g["stringency_index"] >= policy_threshold]
    first_policy = sp["date"].min() if not sp.empty else pd.NaT

    if pd.notna(first_case) and pd.notna(first_policy):
        lag_days = (first_policy - first_case).days
    else:
        lag_days = np.nan

    return first_case, first_policy, lag_days

def compute_country_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-country metrics used in PPEI.
    Returns a DataFrame with columns: country, avg_stringency, mean_log_growth, lag_days
    """
    rows = []
    for country, g in df.groupby("country"):
        g = g.sort_values("date")
        avg_stringency = g["stringency_index"].mean(skipna=True)
        mean_log_growth = g["log_growth"].mean(skipna=True)
        first_case, first_policy, lag_days = compute_first_case_and_first_policy(g)
        rows.append({
            "country": country,
            "avg_stringency": avg_stringency,
            "mean_log_growth": mean_log_growth,
            "lag_days": lag_days
        })
    metrics = pd.DataFrame(rows)
    # sanity: drop countries where we couldn't compute any metrics? keep but warn
    if metrics.empty:
        logger.warning("No country metrics computed (empty input).")
    return metrics

if __name__ == "__main__":
    # quick smoke test (requires pandas)
    logger.info("Running feature_engineering smoke test with dummy data.")
    df = pd.DataFrame({
        "country": ["A", "A", "A", "B", "B"],
        "date": pd.to_datetime(["2020-01-01","2020-01-02","2020-01-08","2020-01-01","2020-01-03"]),
        "stringency_index": [10, 60, 70, 20, 55],
        "cases_per_100k": [0.0, 0.02, 0.05, 0.0, 0.02],
        "log_growth": [0.0, 0.1, 0.2, 0.0, 0.05]
    })
    metrics = compute_country_metrics(df)
    print(metrics)
