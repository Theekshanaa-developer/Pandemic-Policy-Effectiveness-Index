"""
2_ppei_global_analysis/ppei_waves.py
Compute PPEI metrics separately for pandemic waves.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocess import load_data
from utils.feature_engineering import compute_country_metrics
from utils.common import ensure_outdir

sns.set(style="whitegrid")

OUT_DIR = os.path.join("2_ppei_global_analysis", "outputs")
ensure_outdir(OUT_DIR)

# Define pandemic waves
WAVES = {
    "Wave_2020": ("2020-01-01", "2020-12-31"),
    "Wave_2021_Delta": ("2021-01-01", "2021-09-30"),
    "Wave_2021_Omicron": ("2021-10-01", "2022-06-30")
}

def slice_and_compute(df, start, end):
    """Return metrics for a specific wave date range."""
    sliced = df[(df["date"] >= pd.to_datetime(start)) &
                (df["date"] <= pd.to_datetime(end))].copy()
    return compute_country_metrics(sliced)

if __name__ == "__main__":
    df = load_data()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"])

    all_results = []

    for wave_name, (start, end) in WAVES.items():
        print(f"Processing {wave_name}: {start} → {end}")

        metrics = slice_and_compute(df, start, end)
        metrics["wave"] = wave_name

        # Save metrics CSV
        out_csv = os.path.join(OUT_DIR, f"pp_metrics_{wave_name}.csv")
        metrics.to_csv(out_csv, index=False)
        print(f"Saved → {out_csv}")

        # Top 10 (lowest mean log growth = best containment)
        top10 = metrics.sort_values("mean_log_growth").head(10)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=top10, x="mean_log_growth", y="country",
                    palette="viridis", ax=ax)

        ax.set_title(f"Top 10 Countries – {wave_name}")
        fig_path = os.path.join(OUT_DIR, f"top10_{wave_name}.png")
        fig.savefig(fig_path, bbox_inches='tight', dpi=300)

        print(f"Saved → {fig_path}\n")

        all_results.append(metrics)

    # Combined CSV for all waves
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "ppei_by_wave_metrics.csv"), index=False)

    print("Combined wave metrics saved.")
