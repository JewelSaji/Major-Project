"""Shared plotting style helpers for publication-ready figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_publication_style() -> None:
    """Set a clean, journal-friendly plotting style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
        }
    )


def save_publication_figure(fig, path: str) -> None:
    """Consistent save settings across all generated figures."""
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)
