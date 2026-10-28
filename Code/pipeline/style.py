"""Figure sizing for a revtex4-1 two-column paper.

Figures are built at their printed width with fonts near body-text size, so
include them in LaTeX without a width option and nothing gets rescaled.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl

from .config import FIGURE_DIR

COLUMN = 3.40   # \columnwidth in inches (figure)
PAGE = 7.05     # \textwidth in inches (figure*)
DPI = 400


def use_revtex(base=8):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base + 1,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "figure.titlesize": base + 2,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.8,
        "legend.labelspacing": 0.25,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
    })


def save(fig, name):
    FIGURE_DIR.mkdir(exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {path.relative_to(FIGURE_DIR.parent)}")
