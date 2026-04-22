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


def show_biases(pheno):
    cols = [c for c in pheno.columns if c not in ["x", "y", "populations"]]
    k = int(np.sqrt(pheno["populations"].nunique()))
    grid = pheno.groupby(["x", "y"], as_index=False)[cols].mean()

    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 6))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        _plot_grid_heatmap(ax=ax, grid=grid, value_col=col, k=k, title=col, cmap=mpl.cm.viridis)

    plt.tight_layout()
    plt.show()

def visualize_grid_and_pcs(pcs, humans, dpi=200, s=20):
    k = int(np.sqrt(humans["populations"].nunique()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
    ax_grid, ax_pcs = axes

    for i in range(k):
        for j in range(k):
            cell = humans[(humans["x"] == i + 1) & (humans["y"] == j + 1)]
            cell_val = cell["z_outbred"].mean()
            cell_color = map_to_color(i + 1, j + 1, cell_val, humans, "z_outbred")
            ax_grid.add_patch(plt.Rectangle((i, j), 1, 1, facecolor=cell_color, edgecolor="black"))

    ax_grid.set_xlim(0, k)
    ax_grid.set_ylim(0, k)
    ax_grid.set_aspect("equal")
    ax_grid.set_xticks(range(k + 1))
    ax_grid.set_yticks(range(k + 1))
    ax_grid.grid(True)
    ax_grid.set_title("Population grid")

    PC_complete = pd.DataFrame(
        pcs,
        columns=[f"PC{i+1}" for i in range(pcs.shape[1])],
        index=humans.index,
    )

    colors_outbred = [
        map_to_color(x, y, z, humans, "z_outbred")
        for x, y, z in zip(humans["x"], humans["y"], humans["z_outbred"])
    ]

    ax_pcs.scatter(PC_complete["PC1"], PC_complete["PC2"], c=colors_outbred, s=s, linewidths=0)
    ax_pcs.set_aspect("equal", adjustable="box")
    ax_pcs.set_xlabel("PC1")
    ax_pcs.set_ylabel("PC2")
    ax_pcs.set_title("PCs")

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def show_top_snps_ordered(humans, geno, values, k=None):
    values_ord = values.sort_values("neg_log_p", ascending=True).reset_index(drop=True)

    if k is None:
        k = int(max(humans["x"].max(), humans["y"].max()))

    base = humans[["x", "y"]]
    fig, axes = plt.subplots(1, len(values_ord), figsize=(6 * len(values_ord), 6))
    axes = np.atleast_1d(axes)

    seen = {}

    for ax, row in zip(axes, values_ord.itertuples(index=False)):
        snp = row.names
        reason = row.reasons
        metric = row.metric
        negp = row.neg_log_p

        seen[snp] = seen.get(snp, 0) + 1
        title = f"{snp} ({seen[snp]})\n{reason}\n{metric}, neg_log_p={negp:.2g}"

        df = base.copy()
        df["v"] = geno[snp].values
        grid = df.groupby(["x", "y"], as_index=False)["v"].mean()

        _plot_grid_heatmap(ax=ax, grid=grid, value_col="v", k=k, title=title, cmap=mpl.cm.viridis)
        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.show()
