import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as stats
import seaborn as sns

def map_to_color(x, y, z, df, value):
    r = x / df["x"].max() if df["x"].max() != 0 else 0
    g = y / df["y"].max() if df["y"].max() != 0 else 0
    b = z / df[value].max() if df[value].max() != 0 else 0
    return (r, g, b)


def _plot_grid_heatmap(
    ax,
    grid: pd.DataFrame,
    value_col: str,
    k: int,
    title: str = "",
    cmap=mpl.cm.viridis,
    vmin=None,
    vmax=None,
    missing_color="white",
    edgecolor="black",
    linewidth=1.0,
):
    if vmin is None:
        vmin = float(np.nanmin(grid[value_col].to_numpy()))
    if vmax is None:
        vmax = float(np.nanmax(grid[value_col].to_numpy()))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    lookup = {}
    for r in grid.itertuples(index=False):
        lookup[(int(r.x), int(r.y))] = float(getattr(r, value_col))

    for i in range(1, k + 1):
        for j in range(1, k + 1):
            v = lookup.get((i, j), np.nan)
            color = missing_color if np.isnan(v) else cmap(norm(v))
            ax.add_patch(
                plt.Rectangle(
                    (i - 1, j - 1),
                    1,
                    1,
                    facecolor=color,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )
            )

    ax.set_xlim(0, k)
    ax.set_ylim(0, k)
    ax.set_aspect("equal")
    ax.set_xticks(range(k + 1))
    ax.set_yticks(range(k + 1))
    ax.grid(True)
    ax.set_title(title)

def show_biases(pheno, cols):
    k = int(np.sqrt(pheno["populations"].nunique()))
    grid = pheno.groupby(["x", "y"], as_index=False)[cols].mean()

    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 6))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        _plot_grid_heatmap(ax=ax, grid=grid, value_col=col, k=k, title=col, cmap=mpl.cm.viridis)

    plt.tight_layout()
    plt.show()
