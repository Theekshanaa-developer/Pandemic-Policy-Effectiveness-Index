"""
utils/common.py
Common helpers: plotting, saving figures, and small utilities.
"""

import os
import logging
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sns.set(style="whitegrid")

def ensure_outdir(path: str) -> str:
    """Create directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path

def save_fig(fig: plt.Figure, path: str, dpi: int = 300):
    """Save matplotlib figure with tight layout and close it."""
    ensure_outdir(os.path.dirname(path))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)

def quick_barplot(df, x: str, y: str, title: str, outpath: str, figsize=(8,6), palette: Optional[str] = "viridis"):
    """Convenience wrapper to draw a barplot and save to outpath."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x=x, y=y, ax=ax, palette=palette)
    ax.set_title(title)
    save_fig(fig, outpath)
