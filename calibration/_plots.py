"""The reliability diagram shared by ``calibrate.py`` and ``calibrate_from_production.py``."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibration._isotonic_map import GATE

# Reference palette (dataviz skill): observed = blue, old served map = orange, new map = aqua; gate in muted ink.
COLORS = {"observed": "#2a78d6", "old": "#eb6834", "new": "#1baf7a", "gate": "#52514e"}


def plot_reliability(panels, path, xlabel="confidence, bin lower edge", observed_label="observed",
                     new_label="new map"):
    """One panel per verdict: observed precision per confidence bin vs the map's value (and the old map's, if given).

    Args:
        panels: list of ``(title, table)`` where ``table`` is a ``reliability_table`` DataFrame; an optional
            ``old_predicted`` column adds the old map as a third series.
        path: PNG to write.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.8), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (title, tab) in zip(axes, panels):
        ax.axhline(GATE, color=COLORS["gate"], linestyle="--", linewidth=1, label=f"gate {GATE}")
        if len(tab):
            x = np.arange(len(tab))
            ax.plot(x, tab.observed, "o-", color=COLORS["observed"], linewidth=2, markersize=8, label=observed_label)
            if "old_predicted" in tab:
                ax.plot(x, tab.old_predicted, "s--", color=COLORS["old"], linewidth=2, markersize=7,
                        label="old served map")
            ax.plot(x, tab.predicted, "^-", color=COLORS["new"], linewidth=2, markersize=8, label=new_label)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{lo:g}\nn={n:,}" for lo, n in zip(tab.bin_lo, tab.n)], fontsize=7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("precision of the verdict")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
