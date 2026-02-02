# standard library
import os
from pathlib import Path

# third-party
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import scipy.stats as stats  # use stats.linregress, etc.

# local
from scripts.parse_vars import parse_variables, load_config

def map_to_color(x, y, z, df, value):
    # Check if there's variance to avoid division by zero
    r = x / df['x'].max() if df['x'].max() != 0 else 0  # Red component based on 'x'
    g = y / df['y'].max() if df['y'].max() != 0 else 0  # Green component based on 'y'
    b = z / df[value].max() if df[value].max() != 0 else 0  # Blue component based on 'z'
    
    return (r, g, b)

def load_data():
    PATH_VARS = Path("../../geno_simulation.txt")
    R_DIRECTORY = Path("../../rstudio_geno_simulation")
    
    cfg = load_config(PATH_VARS)
    G = int(cfg["G"]); L = int(cfg["L"]); c = int(cfg["c"]); k = int(cfg["k"]); M = float(cfg["M"]);
    # Build prefix pattern for filtering
    prefix = f"G{G}_L{L}_c{c}_k{k}_M{M}"
    F_outbred = 0.0
    path_pheno = f"simulation_data/G{G}_L{L}_c{c}_k{k}_M{M}_F{F_outbred}/phenotype/"
    humans = pd.read_pickle(f"{path_pheno}/humans.pkl")
    
    path_geno = f"simulation_data/G{G}_L{L}_c{c}_k{k}_M{M}_F{F_outbred}/genotype/"
    geno = (pd.read_pickle((f"{path_geno}/complete.pkl"))+1)/2 #(stays between 0 and 1)
    return geno, humans

def make_pheno(humans):
    pheno = humans[['x','y','populations']].copy()
    pheno['no_bias'] = humans['z_outbred']
    pheno['linear'] = pheno['x'] + pheno['y']
    k = int(np.sqrt(len(humans['populations'].unique())))
    # Sinusoidal pattern (e.g., across x)
    freq_x = 3  # 3 full sine cycles across the x-dimension
    freq_y = 2  # 2 full sine cycles across the y-dimension
    
    pheno['sine_x_mix'] = np.round(np.sin(pheno['x'] * freq_x * np.pi / k), 2)
    pheno['sine_y_mix'] = np.round(np.sin(pheno['y'] * freq_y * np.pi / k), 2)
    # Interaction term (e.g., product of x and y)
    pheno['sine_x_y_mix'] = np.round(pheno['sine_x_mix'] + pheno['sine_y_mix'],2)
    n = int(k - k//3)
    pheno['discrete'] = ((pheno['x'] == n) & (pheno['y'] == n)).astype(int)
    return pheno

def show_biases(pheno):


    cols = [c for c in pheno.columns if c not in ["x", "y", "populations"]]

    k = int(np.sqrt(pheno["populations"].nunique()))
    grid = pheno.groupby(["x", "y"], as_index=False)[cols].mean()

    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 6))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):

        # viridis for EVERYTHING (including discrete)
        norm = mpl.colors.Normalize(
            vmin=grid[col].min(),
            vmax=grid[col].max()
        )
        cmap = mpl.cm.viridis

        for i in range(1, k + 1):
            for j in range(1, k + 1):
                v = grid.loc[
                    (grid["x"] == i) & (grid["y"] == j), col
                ].values[0]

                ax.add_patch(
                    plt.Rectangle(
                        (i - 1, j - 1),
                        1,
                        1,
                        facecolor=cmap(norm(v)),
                        edgecolor="black",
                    )
                )

        ax.set_xlim(0, k)
        ax.set_ylim(0, k)
        ax.set_aspect("equal")
        ax.set_xticks(range(k + 1))
        ax.set_yticks(range(k + 1))
        ax.grid(True)
        ax.set_title(col)

    plt.tight_layout()
    plt.show()


def _prep_geno_and_pcs(geno: pd.DataFrame, PCs: int = 5):
    """
    Returns:
      G        : (n, m) numpy array of SNP genotypes (float)
      snp_cols : list of SNP column names
      pcs      : (n, PCs) numpy array of top PCs computed from standardized G
    """
    # assume SNPs are the numeric columns (exclude common metadata if present)
    exclude = {"x", "y", "populations", "population", "pop", "iid", "id"}
    numeric_cols = [c for c in geno.columns if pd.api.types.is_numeric_dtype(geno[c])]
    snp_cols = [c for c in numeric_cols if c not in exclude]

    if len(snp_cols) == 0:
        raise ValueError("No SNP columns found. Ensure geno has numeric SNP columns.")

    G = geno[snp_cols].to_numpy(dtype=float)

    # mean-impute missing genotypes SNP-wise
    if np.isnan(G).any():
        col_means = np.nanmean(G, axis=0)
        inds = np.where(np.isnan(G))
        G[inds] = col_means[inds[1]]

    # standardize SNPs for PCA (avoid divide-by-zero on monomorphic SNPs)
    mu = G.mean(axis=0)
    sd = G.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Gz = (G - mu) / sd

    pca = PCA(n_components=PCs, svd_solver="auto", random_state=0)
    pcs = pca.fit_transform(Gz)

    return G, snp_cols, pcs



def find_snps(geno: pd.DataFrame, pop, chosen_bias, PCs: int = 5):
    y = np.asarray(chosen_bias.values, dtype=float)

    snp_cols = [c for c in geno.columns if pd.api.types.is_numeric_dtype(geno[c])]
    G = geno[snp_cols]

    rows = []
    def add_row(snp, beta, intercept, pval, reason, metric=None, metric_value=None):
        pval = float(pval)
        safe_p = max(pval, np.finfo(float).tiny)
        rows.append({
            "names": snp,
            "betas": float(beta),
            "intercepts": float(intercept),
            "p_vals": pval,
            "neg_log_p": float(-np.log10(safe_p)),
            "reasons": reason,
            "metric": metric,
            "metric_value": None if metric_value is None else float(metric_value),
        })

    hetaf = (G == 0.5).astype(int)
    majaf = (G == 1.0).astype(int)
    minaf = (G == 0.0).astype(int)

    maj_mean = majaf.groupby(pop).mean()
    min_mean = minaf.groupby(pop).mean()
    het_mean = hetaf.groupby(pop).mean()

    maj_mean_id = maj_mean.loc[pop].reset_index(drop=True)
    min_mean_id = min_mean.loc[pop].reset_index(drop=True)
    het_mean_id = het_mean.loc[pop].reset_index(drop=True)

    p = ((maj_mean_id - min_mean_id) + 1) / 2
    p = p.clip(0, 1)
    q = 1 - p
    HWE_dev = het_mean_id - 2 * p * q

    p_pop = ((maj_mean - min_mean) + 1) / 2
    p_pop = p_pop.clip(0, 1)
    p_var = p_pop.var(axis=0)

    max_var_snp_p = p_var.idxmax()
    x = G[max_var_snp_p].values
    beta, intercept, r, pval, stderr = stats.linregress(x, y)
    add_row(max_var_snp_p, beta, intercept, pval, "high var p", "var_p", p_var[max_var_snp_p])

    min_var_snp_p = p_var.idxmin()
    x = G[min_var_snp_p].values
    beta, intercept, r, pval, stderr = stats.linregress(x, y)
    add_row(min_var_snp_p, beta, intercept, pval, "low var p", "var_p", p_var[min_var_snp_p])

    het_var = het_mean.var(axis=0)

    max_var_snp_H = het_var.idxmax()
    x = G[max_var_snp_H].values
    beta, intercept, r, pval, stderr = stats.linregress(x, y)
    add_row(max_var_snp_H, beta, intercept, pval, "high var H", "var_H", het_var[max_var_snp_H])

    min_var_snp_H = het_var.idxmin()
    x = G[min_var_snp_H].values
    beta, intercept, r, pval, stderr = stats.linregress(x, y)
    add_row(min_var_snp_H, beta, intercept, pval, "low var H", "var_H", het_var[min_var_snp_H])

    # GWAS vs chosen_bias
    results = []
    for snp in snp_cols:
        x = G[snp].values
        if np.std(x) == 0:
            continue
        beta, intercept, r, pval, stderr = stats.linregress(x, y)
        results.append((snp, beta, intercept, pval))

    gwas = pd.DataFrame(results, columns=["snp", "beta", "intercept", "p_value"])

    best = gwas.loc[gwas["p_value"].idxmin()]
    add_row(best["snp"], best["beta"], best["intercept"], best["p_value"],
            "lowest pval for bias", "p_value", best["p_value"])

    worst = gwas.loc[gwas["p_value"].idxmax()]
    add_row(worst["snp"], worst["beta"], worst["intercept"], worst["p_value"],
            "highest pval for bias", "p_value", worst["p_value"])

    # PC-based SNP bias
    X = G.to_numpy(dtype=float)
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        rr, cc = np.where(np.isnan(X))
        X[rr, cc] = col_means[cc]

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    pcs = PCA(n_components=PCs, random_state=0).fit_transform(Xz)

    Xc = X - X.mean(axis=0)
    Xsd = Xc.std(axis=0, ddof=0)
    Xsd[Xsd == 0] = np.inf

    pc_scores = np.zeros(X.shape[1], dtype=float)
    for t in range(pcs.shape[1]):
        pc = pcs[:, t] - pcs[:, t].mean()
        pc_sd = pc.std(ddof=0)
        if pc_sd == 0:
            continue
        r = (Xc * pc[:, None]).mean(axis=0) / (Xsd * pc_sd)
        pc_scores = np.maximum(pc_scores, np.abs(r))

    i_max = int(np.argmax(pc_scores))
    i_min = int(np.argmin(pc_scores))
    most_pc_snp = snp_cols[i_max]
    least_pc_snp = snp_cols[i_min]

    x = G[most_pc_snp].values
    beta, intercept, r, pval, stderr = stats.linregress(x, y)
    add_row(most_pc_snp, beta, intercept, pval,
            "most biased to PCs", "max_abs_corr_pc", pc_scores[i_max])

    x = G[least_pc_snp].values
    beta, intercept, r, pval, stderr = stats.linregress(x, y)
    add_row(least_pc_snp, beta, intercept, pval,
            "least biased to PCs", "max_abs_corr_pc", pc_scores[i_min])

    # --- GWAS with PCs as covariates (PC-corrected association) ---
    # Idea: regress out PCs from both y and each SNP, then do simple corr/residual regression.
    # This is equivalent to testing SNP in y ~ SNP + PCs (up to numerical precision).
    
    C = np.column_stack([np.ones(len(y)), pcs])  # intercept + PCs  (n x (1+PCs))
    
    # residualize y on covariates
    coef_y, *_ = np.linalg.lstsq(C, y, rcond=None)
    y_resid = y - C @ coef_y
    
    # residualize all SNPs on covariates in one shot
    # X is (n x m) from your earlier code
    B, *_ = np.linalg.lstsq(C, X, rcond=None)     # (1+PCs) x m
    X_resid = X - C @ B                           # (n x m)
    
    # compute per-SNP Pearson r with residualized y (vectorized)
    yr = y_resid - y_resid.mean()
    xr = X_resid - X_resid.mean(axis=0)
    
    den = (np.sqrt((yr**2).sum()) * np.sqrt((xr**2).sum(axis=0)))
    den[den == 0] = np.inf
    r = (yr[:, None] * xr).sum(axis=0) / den
    r = np.clip(r, -1.0, 1.0)
    
    # convert r -> p-value using t distribution with df = n - (PCs + 2)
    df = len(y) - (PCs + 2)
    t = r * np.sqrt(df / (1 - r**2 + 1e-300))
    pvals_pc = 2 * stats.t.sf(np.abs(t), df)
    
    # pick best/worst after PC correction
    i_best = int(np.argmin(pvals_pc))
    i_worst = int(np.argmax(pvals_pc))
    
    best_snp_pc = snp_cols[i_best]
    worst_snp_pc = snp_cols[i_worst]
    
    # for reporting beta/intercept comparable to your table:
    # compute slope in residual space: beta = cov(x_resid, y_resid) / var(x_resid)
    xv = X_resid[:, i_best]
    beta_best = np.cov(xv, y_resid, ddof=0)[0, 1] / (np.var(xv, ddof=0) + 1e-300)
    add_row(best_snp_pc, beta_best, 0.0, pvals_pc[i_best],
            reason="lowest pval for bias (PC-corrected)",
            metric="p_value_pc_corrected", metric_value=pvals_pc[i_best])
    
    xv = X_resid[:, i_worst]
    beta_worst = np.cov(xv, y_resid, ddof=0)[0, 1] / (np.var(xv, ddof=0) + 1e-300)
    add_row(worst_snp_pc, beta_worst, 0.0, pvals_pc[i_worst],
            reason="highest pval for bias (PC-corrected)",
            metric="p_value_pc_corrected", metric_value=pvals_pc[i_worst])
    
    # return: table + useful arrays
    return pd.DataFrame(rows), pcs, p, HWE_dev

def visualize_grid_and_pcs(pcs, humans, dpi=200, s=20):


    k = int(np.sqrt(humans["populations"].nunique()))

    # EXACTLY like show_biases: two columns, each 6x6
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
    ax_grid, ax_pcs = axes

    # =========================
    # 1) POPULATION GRID (same as bias plots)
    # =========================
    for i in range(k):
        for j in range(k):
            cell = humans[(humans["x"] == i + 1) & (humans["y"] == j + 1)]
            cell_val = cell["z_outbred"].mean()
            cell_color = map_to_color(i + 1, j + 1, cell_val, humans, "z_outbred")

            ax_grid.add_patch(
                plt.Rectangle((i, j), 1, 1,
                              facecolor=cell_color,
                              edgecolor="black")
            )

    ax_grid.set_xlim(0, k)
    ax_grid.set_ylim(0, k)
    ax_grid.set_aspect("equal")
    ax_grid.set_xticks(range(k + 1))
    ax_grid.set_yticks(range(k + 1))
    ax_grid.grid(True)
    ax_grid.set_title("Population grid")

    # =========================
    # 2) PCS (same-sized panel)
    # =========================
    PC_complete = pd.DataFrame(
        pcs,
        columns=[f"PC{i+1}" for i in range(pcs.shape[1])],
        index=humans.index
    )

    colors_outbred = [
        map_to_color(x, y, z, humans, "z_outbred")
        for x, y, z in zip(humans["x"], humans["y"], humans["z_outbred"])
    ]

    ax_pcs.scatter(
        PC_complete["PC1"],
        PC_complete["PC2"],
        c=colors_outbred,
        s=s,
        linewidths=0
    )

    ax_pcs.set_aspect("equal", adjustable="box")
    ax_pcs.set_xlabel("PC1")
    ax_pcs.set_ylabel("PC2")
    ax_pcs.set_title("PCs")

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def show_top_snps_ordered(humans, geno, values, k=None):


    # order by neg_log_p (low → high)
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

        title = (
            f"{snp} ({seen[snp]})\n"
            f"{reason}\n"
            f"{metric}, neg_log_p={negp:.2g}"
        )

        df = base.copy()
        df["v"] = geno[snp]

        grid = df.groupby(["x", "y"], as_index=False)["v"].mean()

        norm = mpl.colors.Normalize(
            vmin=grid["v"].min(),
            vmax=grid["v"].max()
        )
        cmap = mpl.cm.viridis

        for i in range(1, k + 1):
            for j in range(1, k + 1):
                v = grid.loc[
                    (grid["x"] == i) & (grid["y"] == j), "v"
                ].values[0]

                ax.add_patch(
                    plt.Rectangle(
                        (i - 1, j - 1),
                        1,
                        1,
                        facecolor=cmap(norm(v)),
                        edgecolor="black",
                    )
                )

        ax.set_xlim(0, k)
        ax.set_ylim(0, k)
        ax.set_aspect("equal")
        ax.set_xticks(range(k + 1))
        ax.set_yticks(range(k + 1))
        ax.grid(True)
        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.show()


def polygenic_noise(
    geno,
    chosen_snp,
    p_causal=0.01,
    n_causal=None,
    total_beta=0.2,
    seed=1,
    downweight_by_maf_sd=True,
    regen=False,
    cache_path="g_noise.npz",
):
    """
    If regen=False and cache_path exists, load and return cached (g_noise, beta, is_causal).
    Otherwise generate, save to cache_path, and return.

    Cache format: npz with arrays: g_noise, beta, is_causal and meta: chosen_snp, seed, p_causal, n_causal, total_beta.
    """

    # ---- load cache if allowed ----
    if (not regen) and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return d["g_noise"], d["beta"], d["is_causal"].astype(bool)

    # ---- generate ----
    cols = list(geno.columns)
    assert chosen_snp in cols, f"{chosen_snp} not in geno.columns"

    G = geno.to_numpy(dtype=np.float32)
    N, M = G.shape

    j = cols.index(chosen_snp)
    rng = np.random.default_rng(seed)

    # choose causal SNPs from all indices except chosen_snp
    eligible = np.arange(M, dtype=int)
    eligible = eligible[eligible != j]

    if n_causal is not None:
        n_causal = int(n_causal)
        n_causal = max(0, min(n_causal, eligible.size))
        causal_idx = rng.choice(eligible, size=n_causal, replace=False)
    else:
        p_causal = float(p_causal)
        p_causal = max(0.0, min(p_causal, 1.0))
        n_causal_eff = int(np.round(p_causal * eligible.size))
        causal_idx = rng.choice(eligible, size=n_causal_eff, replace=False)

    is_causal = np.zeros(M, dtype=bool)
    is_causal[causal_idx] = True

    beta = np.zeros(M, dtype=np.float32)
    beta[is_causal] = rng.normal(0.0, 1.0, size=is_causal.sum()).astype(np.float32)

    if downweight_by_maf_sd:
        maf = np.array([float(c.split("_MAF_")[-1]) for c in cols], dtype=np.float32)
        sd = np.sqrt(2.0 * maf * (1.0 - maf))
        sd[sd == 0] = 1.0
        beta = beta / sd

    beta[j] = 0.0  # ensure chosen SNP excluded from polygenic term

    # scale to desired total magnitude (L2 norm)
    norm = float(np.linalg.norm(beta))
    if norm > 0:
        beta *= (total_beta / norm)

    g_noise = G @ beta
    g_noise = (g_noise - g_noise.mean()) / (g_noise.std() + 1e-8)

    # ---- save cache (npz, not pickle) ----
    np.savez_compressed(
        cache_path,
        g_noise=g_noise.astype(np.float32),
        beta=beta.astype(np.float32),
        is_causal=is_causal.astype(np.uint8),
        chosen_snp=np.array(chosen_snp),
        seed=np.array(seed),
        p_causal=np.array(p_causal),
        n_causal=np.array(-1 if n_causal is None else int(n_causal)),
        total_beta=np.array(total_beta),
    )

    return g_noise, beta, is_causal


def plot_effects(pheno, effects):


    cols = ["snp", "pop", "poly", "total"]

    # grid size
    k = int(np.sqrt(pheno["populations"].nunique()))

    # attach effects to pheno (index-aligned)
    pheno = pheno.copy()
    pheno[cols] = effects[cols].to_numpy()

    # one value per population cell
    grid = pheno.groupby(["x", "y"], as_index=False)[cols].mean()

    # consistent scaling across plots (per-column)
    vmins = {c: grid[c].min() for c in cols}
    vmaxs = {c: grid[c].max() for c in cols}

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        norm = mpl.colors.Normalize(vmin=vmins[col], vmax=vmaxs[col])
        cmap = mpl.cm.viridis

        for i in range(1, k + 1):
            for j in range(1, k + 1):
                v = grid.loc[(grid["x"] == i) & (grid["y"] == j), col].values[0]
                ax.add_patch(
                    plt.Rectangle(
                        (i - 1, j - 1),
                        1, 1,
                        facecolor=cmap(norm(v)),
                        edgecolor="black",
                    )
                )

        ax.set_xlim(0, k)
        ax.set_ylim(0, k)
        ax.set_aspect("equal")
        ax.set_xticks(range(k + 1))
        ax.set_yticks(range(k + 1))
        ax.grid(True)
        ax.set_title(col)

    plt.tight_layout()
    plt.show()

def plot_components_vs_snp(geno, chosen_snp, effects, jitter=0.03):

    x = geno[chosen_snp].to_numpy(dtype=float)

    ys = {
        "snp":   effects["snp"].to_numpy(dtype=float),
        "pop":   effects["pop"].to_numpy(dtype=float),
        "poly":  effects["poly"].to_numpy(dtype=float),
        "total": effects["total"].to_numpy(dtype=float),
    }

    X = np.column_stack([x] + [ys[k] for k in ys])
    X = X[~np.isnan(X).any(axis=1)]
    x = X[:, 0]
    ys = {k: X[:, i+1] for i, k in enumerate(ys)}

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
        "snp":   "#c45508",
        "pop":   "#1f3a5f",
        "poly":  "#f2c14e",
        "total": "#000000",
    }

    order = ["snp", "pop", "poly", "total"]

    plt.figure(figsize=(7.5, 5.5))

    rng = np.random.default_rng(0)
    plt.scatter(
        x + rng.normal(0, jitter, size=len(x)),
        ys["total"],
        alpha=0.25,
        s=18,
        color="0.5",
    )

    for g in np.unique(x):
        mask = (x == g)
        mean = ys["total"][mask].mean()
        sem = ys["total"][mask].std(ddof=1) / np.sqrt(mask.sum())
        plt.errorbar(
            [g], [mean], yerr=[sem],
            fmt="o", capsize=4,
            color=colors["total"],
        )

    mean_sem_handle = Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markersize=7,
        markerfacecolor="none",
        markeredgewidth=1.5,
        color="black",
        label="mean ± SEM (total)",
    )

    x_line = np.array([x.min(), x.max()])

    # collect betas/intercepts here
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

