import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA

def _get_snp_cols(geno: pd.DataFrame):
    exclude = {"x", "y", "populations", "population", "pop", "iid", "id"}
    numeric_cols = [c for c in geno.columns if pd.api.types.is_numeric_dtype(geno[c])]
    snp_cols = [c for c in numeric_cols if c not in exclude]
    if len(snp_cols) == 0:
        raise ValueError("No SNP columns found. Ensure geno has numeric SNP columns.")
    return snp_cols
def _prep_geno_and_pcs(geno: pd.DataFrame, PCs: int = 5):
    snp_cols = _get_snp_cols(geno)
    G = geno[snp_cols].to_numpy(dtype=float)

    # mean-impute missing genotypes SNP-wise
    if np.isnan(G).any():
        col_means = np.nanmean(G, axis=0)
        rr, cc = np.where(np.isnan(G))
        G[rr, cc] = col_means[cc]

    # standardize SNPs for PCA
    mu = G.mean(axis=0)
    sd = G.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Gz = (G - mu) / sd

    pca = PCA(n_components=PCs, svd_solver="auto", random_state=0)
    pcs = pca.fit_transform(Gz)
    return G, snp_cols, pcs

def per_genotype_freq_per_human(geno: pd.DataFrame, pop):
    # denominator: non-missing genotype count per pop per SNP
    n = geno.notna().groupby(pop).sum()

    # AA (coded 1)
    maj = (geno == 1).astype(float)          # keeps NaN as False -> 0, so use n from geno
    maj_by_pop = maj.groupby(pop).sum() / n
    maj_per_human = maj_by_pop.loc[pd.Index(pop)].set_index(geno.index)

    # aa (coded -1)
    min_ = (geno == -1).astype(float)
    min_by_pop = min_.groupby(pop).sum() / n
    min_per_human = min_by_pop.loc[pd.Index(pop)].set_index(geno.index)

    # Aa (coded 0)
    het = (geno == 0).astype(float)
    het_by_pop = het.groupby(pop).sum() / n
    het_per_human = het_by_pop.loc[pd.Index(pop)].set_index(geno.index)

    return maj_per_human, het_per_human, min_per_human

def top_correlated_snps(
    raw_geno: pd.DataFrame,
    focal_snp: str,
    n: int = 20,
    method: str = "pearson",
    absolute: bool = True,
    least: bool = False,
    drop_self: bool = True,
) -> pd.Series:
    """
    Return the top-n most or least correlated SNPs with `focal_snp`.
    """

    if focal_snp not in raw_geno.columns:
        raise KeyError(f"focal_snp '{focal_snp}' not found in raw_geno.columns")

    if n <= 0:
        raise ValueError("n must be a positive integer")

    focal = raw_geno[focal_snp]
    corrs = raw_geno.corrwith(focal, method=method)

    if drop_self and focal_snp in corrs.index:
        corrs = corrs.drop(index=focal_snp)

    corrs = corrs.dropna()

    if absolute:
        ranking_values = corrs.abs()
    else:
        ranking_values = corrs

    # 🔹 core addition
    ranked_index = (
        ranking_values.sort_values(ascending=least).index
        if absolute
        else ranking_values.sort_values(ascending=least).index
    )

    ranked = corrs.loc[ranked_index]

    return ranked.head(n)

def find_snps(geno: pd.DataFrame, pop, chosen_bias, PCs: int = 5):
    y = np.asarray(chosen_bias.values, dtype=float)

    G_arr, snp_cols, pcs = _prep_geno_and_pcs(geno, PCs=PCs)
    G = pd.DataFrame(G_arr, columns=snp_cols, index=geno.index)

    rows = []

    def add_row(snp, beta, intercept, pval, reason, metric=None, metric_value=None):
        pval = float(pval)
        safe_p = max(pval, np.finfo(float).tiny)
        rows.append(
            {
                "names": snp,
                "betas": float(beta),
                "intercepts": float(intercept),
                "p_vals": pval,
                "neg_log_p": float(-np.log10(safe_p)),
                "reasons": reason,
                "metric": metric,
                "metric_value": None if metric_value is None else float(metric_value),
            }
        )

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
    add_row(best["snp"], best["beta"], best["intercept"], best["p_value"], "lowest pval for bias", "p_value", best["p_value"])

    worst = gwas.loc[gwas["p_value"].idxmax()]
    add_row(worst["snp"], worst["beta"], worst["intercept"], worst["p_value"], "highest pval for bias", "p_value", worst["p_value"])

    # PC-correlation score (reuse pcs from _prep_geno_and_pcs)
    X = G_arr
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
    add_row(most_pc_snp, beta, intercept, pval, "most biased to PCs", "max_abs_corr_pc", pc_scores[i_max])

    x = G[least_pc_snp].values
    beta, intercept, r, pval, stderr = stats.linregress(x, y)
    add_row(least_pc_snp, beta, intercept, pval, "least biased to PCs", "max_abs_corr_pc", pc_scores[i_min])

    # PC-corrected association
    C = np.column_stack([np.ones(len(y)), pcs])

    coef_y, *_ = np.linalg.lstsq(C, y, rcond=None)
    y_resid = y - C @ coef_y

    B, *_ = np.linalg.lstsq(C, X, rcond=None)
    X_resid = X - C @ B

    yr = y_resid - y_resid.mean()
    xr = X_resid - X_resid.mean(axis=0)

    den = (np.sqrt((yr**2).sum()) * np.sqrt((xr**2).sum(axis=0)))
    den[den == 0] = np.inf

    r = (yr[:, None] * xr).sum(axis=0) / den
    r = np.clip(r, -1.0, 1.0)

    df = len(y) - (PCs + 2)
    t = r * np.sqrt(df / (1 - r**2 + 1e-300))
    pvals_pc = 2 * stats.t.sf(np.abs(t), df)

    i_best = int(np.argmin(pvals_pc))
    i_worst = int(np.argmax(pvals_pc))

    best_snp_pc = snp_cols[i_best]
    worst_snp_pc = snp_cols[i_worst]

    xv = X_resid[:, i_best]
    beta_best = np.cov(xv, y_resid, ddof=0)[0, 1] / (np.var(xv, ddof=0) + 1e-300)
    add_row(best_snp_pc, beta_best, 0.0, pvals_pc[i_best], "lowest pval for bias (PC-corrected)", "p_value_pc_corrected", pvals_pc[i_best])

    xv = X_resid[:, i_worst]
    beta_worst = np.cov(xv, y_resid, ddof=0)[0, 1] / (np.var(xv, ddof=0) + 1e-300)
    add_row(worst_snp_pc, beta_worst, 0.0, pvals_pc[i_worst], "highest pval for bias (PC-corrected)", "p_value_pc_corrected", pvals_pc[i_worst])

    return pd.DataFrame(rows), pcs, p, HWE_dev

def get_ld_partners(raw_geno, chosen_snp, r2_threshold=0.1, window=200):
    """Find SNPs in LD with chosen_snp within a column window."""
    chosen_idx = raw_geno.columns.get_loc(chosen_snp)
    window_cols = raw_geno.columns[
        max(0, chosen_idx - window) : min(len(raw_geno.columns), chosen_idx + window)
    ]

    chosen_g = raw_geno[chosen_snp].values.astype(float)
    ld_partners = []

    for col in window_cols:
        if col == chosen_snp:
            continue
        g = raw_geno[col].values.astype(float)
        # r² between chosen and candidate
        mask = ~(np.isnan(chosen_g) | np.isnan(g))
        r = np.corrcoef(chosen_g[mask], g[mask])[0, 1]
        if r**2 >= r2_threshold:
            ld_partners.append((col, r**2))

    return pd.DataFrame(ld_partners, columns=["snp", "r2"]).sort_values(
        "r2", ascending=False
    )
    
def add_genetic_effects(
    raw_geno,
    pheno,
    chosen_snp,
    h2_causal=0.3,  # total heritability from causal SNPs
    h2_chosen=0.15,  # fraction going to chosen_snp specifically
    h2_ld=0.08,  # fraction going to LD partners
    n_random=20,  # number of random causal SNPs
    r2_threshold=0.1,
    seed=42,
):

    rng = np.random.default_rng(seed)

    # --- standardize helper ---
    def std(g):
        mu, sigma = g.mean(), g.std()
        return (g - mu) / (sigma + 1e-8)

    prs = np.zeros(len(raw_geno))
    causal_snps = {}

    # 1. Chosen SNP — largest direct effect
    beta_chosen = np.sqrt(h2_chosen)
    prs += beta_chosen * std(raw_geno[chosen_snp].values.astype(float))
    causal_snps[chosen_snp] = beta_chosen

    # 2. LD partners — effect decays with r²
    ld = get_ld_partners(raw_geno, chosen_snp, r2_threshold=r2_threshold)
    if len(ld) > 0:
        # distribute h2_ld proportional to r²
        ld["weight"] = ld["r2"] / ld["r2"].sum()
        for _, row in ld.iterrows():
            beta = np.sqrt(h2_ld * row["weight"]) * rng.choice([-1, 1])
            prs += beta * std(raw_geno[row["snp"]].values.astype(float))
            causal_snps[row["snp"]] = beta

    # 3. Random causal SNPs — small polygenic background
    h2_random = h2_causal - h2_chosen - h2_ld
    exclude = set(causal_snps.keys())
    candidates = [c for c in raw_geno.columns if c not in exclude]
    random_snps = rng.choice(candidates, size=n_random, replace=False)
    beta_random = rng.normal(0, np.sqrt(h2_random / n_random), size=n_random)
    for snp, beta in zip(random_snps, beta_random):
        prs += beta * std(raw_geno[snp].values.astype(float))
        causal_snps[snp] = beta

    # 4. Standardize PRS to unit variance, then scale by sqrt(h2_causal)
    prs = std(prs) * np.sqrt(h2_causal)

    # 5. Add to each phenotype bias + environmental noise
    results = {}
    for bias_name in pheno.columns:

        bias = std(pheno[bias_name].values.astype(float))
        h2_bias = 0.3  # how much variance comes from stratification

        noise = rng.normal(0, np.sqrt(1 - h2_causal - h2_bias), size=len(raw_geno))

        y = np.sqrt(h2_causal) * std(prs) + np.sqrt(h2_bias) * bias + noise

        results[f"pheno_{bias_name}"] = y

    return pd.DataFrame(results, index=pheno.index), causal_snps