"""
utils/preprocess.py
Data loading and basic preprocessing utilities for the PPEI project.
"""

import os
import logging
from typing import List, Optional

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
DEFAULT_RAW_DIR = os.path.join("data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join("data", "processed")
DEFAULT_INPUT = os.path.join(DEFAULT_RAW_DIR, "OxCGRT_cleaned_normalized.csv")
# ----------------------------------------

def load_data(path: str = DEFAULT_INPUT, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame. parse_dates defaults to ['date'] if available."""
    if parse_dates is None:
        parse_dates = ["date"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path, parse_dates=parse_dates)
    return df

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and check required columns exist."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    required = ["country", "date", "stringency_index", "new_cases", "population"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    return df

def add_per100k_and_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cases_per_100k, 7-day rolling average (cases_7d), and log_growth columns.
    Returns a new DataFrame (does not modify in place).
    """
    df = df.copy()
    # compute cases per 100k
    df["cases_per_100k"] = df["new_cases"] / df["population"] * 100000
    df = df.sort_values(["country", "date"])
    # 7-day rolling mean of cases_per_100k per country
    df["cases_7d"] = df.groupby("country")["cases_per_100k"]\
                       .rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    # previous day's smoothed cases
    df["cases_shift"] = df.groupby("country")["cases_7d"].shift(1)
    # log growth (small epsilon to avoid log(0))
    df["log_growth"] = np.log((df["cases_7d"].fillna(0) + 1) / (df["cases_shift"].fillna(0) + 1))
    return df

def save_df(df: pd.DataFrame, path: str):
    """Save dataframe to CSV, creating dirs if necessary."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved: %s", path)

def preprocess_and_save(input_path: str = DEFAULT_INPUT,
                        out_path: str = os.path.join(DEFAULT_PROCESSED_DIR, "panel_preprocessed.csv")) -> pd.DataFrame:
    """Convenience: load, ensure columns, add features, and save processed CSV."""
    df = load_data(input_path)
    df = ensure_columns(df)
    df = add_per100k_and_smoothing(df)
    save_df(df, out_path)
    return df

if __name__ == "__main__":
    # simple CLI behavior: run with default paths
    try:
        df_out = preprocess_and_save()
        logger.info("Preprocessing complete. Rows: %d", len(df_out))
    except Exception as e:
        logger.exception("Preprocessing failed: %s", e)
        raise
