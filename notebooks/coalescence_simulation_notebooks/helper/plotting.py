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


def plot_effects(pheno, effects):
    cols = ["snp", "pop", "poly", "total"]
    k = int(np.sqrt(pheno["populations"].nunique()))

    pheno = pheno.copy()
    pheno[cols] = effects[cols].to_numpy()
    grid = pheno.groupby(["x", "y"], as_index=False)[cols].mean()

    vmins = {c: grid[c].min() for c in cols}
    vmaxs = {c: grid[c].max() for c in cols}

    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 6))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        _plot_grid_heatmap(
            ax=ax,
            grid=grid,
            value_col=col,
            k=k,
            title=col,
            cmap=mpl.cm.viridis,
            vmin=vmins[col],
            vmax=vmaxs[col],
        )

    plt.tight_layout()
    plt.show()


def plot_components_vs_snp(geno, chosen_snp, effects, jitter=0.03):
    x = geno[chosen_snp].to_numpy(dtype=float)

    ys = {
        "snp": effects["snp"].to_numpy(dtype=float),
        "pop": effects["pop"].to_numpy(dtype=float),
        "poly": effects["poly"].to_numpy(dtype=float),
        "total": effects["total"].to_numpy(dtype=float),
    }

    X = np.column_stack([x] + [ys[k] for k in ys])
    X = X[~np.isnan(X).any(axis=1)]
    x = X[:, 0]
    ys = {k: X[:, i + 1] for i, k in enumerate(ys)}

    def fit_line(x, y):
        slope, intercept, r, p, se = stats.linregress(x, y)
        neglogp = -np.log10(p) if p > 0 else np.inf
        return slope, intercept, neglogp

    def pretty(name, slope, intercept, neglogp):
        ptxt = "∞" if np.isinf(neglogp) else f"{neglogp:.2f}"
        sign = "+" if intercept >= 0 else "−"
        itxt = f"{abs(intercept):.3f}"
        return f"{name}: ŷ={slope:.3f}x {sign} {itxt}, −log10(p)={ptxt}"

    colors = {
        "snp": "#c45508",
        "pop": "#1f3a5f",
        "poly": "#f2c14e",
        "total": "#000000",
    }
    order = ["snp", "pop", "poly", "total"]

    plt.figure(figsize=(7.5, 5.5))

    rng = np.random.default_rng(0)
    plt.scatter(x + rng.normal(0, jitter, size=len(x)), ys["total"], alpha=0.25, s=18, color="0.5")

    for g in np.unique(x):
        mask = x == g
        mean = ys["total"][mask].mean()
        sem = ys["total"][mask].std(ddof=1) / np.sqrt(mask.sum())
        plt.errorbar([g], [mean], yerr=[sem], fmt="o", capsize=4, color=colors["total"])

    mean_sem_handle = Line2D(
        [0], [0],
        marker="o", linestyle="none",
        markersize=7,
        markerfacecolor="none",
        markeredgewidth=1.5,
        color="black",
        label="mean ± SEM (total)",
    )

    x_line = np.array([x.min(), x.max()])
    params = {}

    for name in order:
        slope, intercept, neglogp = fit_line(x, ys[name])
        params[name] = {"beta": slope, "intercept": intercept, "neglogp": neglogp}

        plt.plot(
            x_line,
            intercept + slope * x_line,
            linewidth=3 if name == "total" else 2.5,
            color=colors[name],
            label=pretty(name, slope, intercept, neglogp),
            zorder=3 if name == "total" else 2,
        )

    plt.xticks(sorted(np.unique(x)))
    plt.xlabel(f"Genotype at {chosen_snp}")
    plt.ylabel("Effect / phenotype")
    plt.title("snp + pop + jitter = total")

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(mean_sem_handle)
    labels.append(mean_sem_handle.get_label())
    plt.legend(handles, labels, frameon=False, fontsize=8)

    plt.tight_layout()
    plt.show()
    return params


def show_corr_snps_ordered(humans, geno, values, k=None):
    base = humans[["x", "y"]]

    fig, axes = plt.subplots(1, len(values), figsize=(6 * len(values), 6))
    axes = np.atleast_1d(axes)

    for ax, row in zip(axes, values.itertuples(index=False)):
        snp = row.names

        df = base.copy()
        df["v"] = geno[snp].values
        grid = df.groupby(["x", "y"], as_index=False)["v"].mean()

        kk = int(max(grid["x"].max(), grid["y"].max())) if k is None else int(k)

        _plot_grid_heatmap(
            ax=ax,
            grid=grid,
            value_col="v",
            k=kk,
            title=snp,
            cmap=mpl.cm.viridis,
            missing_color="white",
        )
        ax.set_title(snp, fontsize=9)

    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def gwas_pca_beta_pcscan_plot(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_pcs: int = 20,
    plot_top_k: int = 2,
    title: str = "Top PCs explaining GWAS betas",
    palette: str = "coolwarm",
    point_size: int = 18,
    standardize_for_pca: bool = True,
):
    """
    Pipeline:
      1) GWAS: run y ~ SNP for each SNP (column in X) -> beta per SNP
      2) PCA on X.T: get PC scores per SNP (rows=SNPs)
      3) For each PCj: regress beta ~ PCj -> slope/intercept/r/p
      4) Pick best PCs (lowest p) and plot PCa vs PCb with beta as hue

    Parameters
    ----------
    X : pd.DataFrame
        Genotype-like matrix (n_samples x n_snps). Columns are SNP names.
    y : array-like
        Phenotype vector length n_samples.
    n_pcs : int
        Number of PCs computed on X.T (SNP space).
    plot_top_k : int
        How many top PCs to pick for plotting. If 2, plots best PC vs second-best PC.
    standardize_for_pca : bool
        If True, StandardScaler(with_mean=True, with_std=True) is applied to X.T before PCA.

    Returns
    -------
    out : dict
        - gwas : DataFrame with snp, beta, intercept, neglog10p
        - pc_scores : DataFrame indexed by SNP with columns PC1..PCn
        - pc_beta : DataFrame with PC1..PCn + beta (aligned SNPs)
        - pc_assoc : DataFrame with per-PC regression stats vs beta
        - best_pcs : list of selected PC names (length plot_top_k, clipped to n_pcs)
        - pca_model : fitted PCA
        - scaler : fitted StandardScaler or None
    """
    # ---- validate ----
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame (samples x snps).")
    y = np.asarray(y, dtype=float)
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"y has length {y.shape[0]} but X has {X.shape[0]} rows.")

    cols = list(X.columns)

    # ---- 1) GWAS: beta per SNP ----
    betas = np.empty(len(cols), dtype=float)
    intercepts = np.empty(len(cols), dtype=float)
    neglogps = np.empty(len(cols), dtype=float)

    for j, snp in enumerate(cols):
        x = X[snp].to_numpy(dtype=float)
        slope, intercept, r, p, se = stats.linregress(x, y)
        betas[j] = slope
        intercepts[j] = intercept
        neglogps[j] = -np.log10(p) if p > 0 else np.inf

    gwas = pd.DataFrame(
        {"snp": cols, "beta": betas, "intercept": intercepts, "neglog10p": neglogps}
    )

    # ---- 2) PCA on SNPs: X.T is (n_snps x n_samples) ----
    X_t = X.T.to_numpy(dtype=float)

    scaler = None
    if standardize_for_pca:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_t = scaler.fit_transform(X_t)

    pca = PCA(n_components=min(n_pcs, X_t.shape[1], X_t.shape[0]), random_state=0)
    pcs = pca.fit_transform(X_t)

    pc_cols = [f"PC{i}" for i in range(1, pcs.shape[1] + 1)]
    pc_scores = pd.DataFrame(pcs, columns=pc_cols, index=cols)  # indexed by SNP

    # ---- align PCs with betas ----
    pc_beta = pc_scores.join(gwas.set_index("snp")[["beta"]], how="inner")

    # ---- 3) regress beta ~ PCj for each PC ----
    assoc_rows = []
    for pc in pc_cols:
        slope, intercept, r, p, se = stats.linregress(pc_beta[pc].to_numpy(), pc_beta["beta"].to_numpy())
        assoc_rows.append(
            {
                "PC": pc,
                "slope": float(slope),
                "intercept": float(intercept),
                "r": float(r),
                "p": float(p),
                "neglog10p": float(-np.log10(p) if p > 0 else np.inf),
            }
        )

    pc_assoc = pd.DataFrame(assoc_rows).sort_values("p", ascending=True).reset_index(drop=True)

    # ---- 4) pick best PCs and plot ----
    k = min(plot_top_k, len(pc_cols))
    best_pcs = pc_assoc["PC"].head(k).tolist()

    if k >= 2:
        pc_x, pc_y = best_pcs[0], best_pcs[1]
        df = pc_beta.sort_values("beta")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x=pc_x,
            y=pc_y,
            hue="beta",
            palette=palette,
            linewidth=0,
            s=point_size,
        )
        plt.title(f"{title}\n({pc_x} vs {pc_y})")
        plt.xlabel(pc_x)
        plt.ylabel(pc_y)
        plt.legend(title="Beta", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    return {
        "gwas": gwas,
        "pc_scores": pc_scores,
        "pc_beta": pc_beta,
        "pc_assoc": pc_assoc,
        "best_pcs": best_pcs,
        "pca_model": pca,
        "scaler": scaler,
    }
