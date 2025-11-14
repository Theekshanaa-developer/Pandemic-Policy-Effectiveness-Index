"""
2_ppei_global_analysis/ppei_compute.py
Compute PPEI across all countries using standardized metrics and save results.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.preprocess import load_data
from utils.feature_engineering import compute_country_metrics
from utils.common import ensure_outdir

OUT_DIR = os.path.join("2_ppei_global_analysis", "outputs")
ensure_outdir(OUT_DIR)

# Weights used in final composite PPEI score
WEIGHTS = {
    "stringency": 0.3,
    "growth": 0.5,
    "lag": 0.2
}

def compute_ppei(metrics_df, weights=WEIGHTS):
    """Compute PPEI using weighted z-score normalization."""
    m = metrics_df.copy()

    # Features to include
    m["feat_stringency"] = m["avg_stringency"]                     # higher is better
    m["feat_growth"] = -m["mean_log_growth"]                       # lower growth → higher score
    m["feat_lag"] = -m["lag_days"]                                 # lower lag → higher score

    # Fill missing values
    for col in ["feat_stringency", "feat_growth", "feat_lag"]:
        m[col] = m[col].fillna(m[col].median())

    # Standardize features
    scaler = StandardScaler()
    m[["z_stringency", "z_growth", "z_lag"]] = scaler.fit_transform(
        m[["feat_stringency", "feat_growth", "feat_lag"]]
    )

    # Weighted score
    m["raw_score"] = (
        weights["stringency"] * m["z_stringency"]
        + weights["growth"] * m["z_growth"]
        + weights["lag"] * m["z_lag"]
    )

    # Normalize 0–100
    mm = MinMaxScaler(feature_range=(0, 100))
    m["PPEI"] = mm.fit_transform(m[["raw_score"]])

    return m


if __name__ == "__main__":
    # Load full dataset
    df = load_data()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df.sort_values(["country", "date"])

    # Compute metrics per country
    metrics = compute_country_metrics(df)

    # Compute PPEI
    ppei = compute_ppei(metrics)

    # Save
    out_csv = os.path.join(OUT_DIR, "ppei_results.csv")
    ppei.to_csv(out_csv, index=False)

    print(f"PPEI results saved → {out_csv}")
    print("\nTop 10 countries:")
    print(ppei.sort_values("PPEI", ascending=False).head(10)[["country", "PPEI"]])
