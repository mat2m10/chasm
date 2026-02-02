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
    if (not regen) and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return d["g_noise"], d["beta"], d["is_causal"].astype(bool)

    cols = _get_snp_cols(geno)
    assert chosen_snp in cols, f"{chosen_snp} not in geno SNP columns"

    geno = geno[cols]
    G = geno.to_numpy(dtype=np.float32)
    N, M = G.shape

    j = cols.index(chosen_snp)
    rng = np.random.default_rng(seed)

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

    beta[j] = 0.0

    norm = float(np.linalg.norm(beta))
    if norm > 0:
        beta *= (total_beta / norm)

    g_noise = G @ beta
    g_noise = (g_noise - g_noise.mean()) / (g_noise.std() + 1e-8)

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


def standardize_and_return_params(geno):
    cols = _get_snp_cols(geno)
    geno = geno[cols]

    snp_mean = geno.mean(axis=0)
    snp_std = geno.std(axis=0, ddof=0)
    geno_std = (geno - snp_mean) / snp_std

    stats_df = pd.DataFrame({"snp": geno.columns, "mean": snp_mean.values, "std": snp_std.values})
    return geno_std, stats_df


def snp_correlation_analysis(geno_std, chosen_snp, top_k=5):
    X = geno_std.values
    corr = np.corrcoef(X, rowvar=False).astype(np.float32)
    np.fill_diagonal(corr, 0.0)

    mean_abs_corr = np.mean(np.abs(corr), axis=0)
    mean_abs_corr = pd.Series(mean_abs_corr, index=geno_std.columns)

    pearson_corr = geno_std.corrwith(geno_std[chosen_snp])
    corr_df = pearson_corr.reset_index()
    corr_df.columns = ["snp", "pearson_r"]
    corr_df["abs_r"] = corr_df["pearson_r"].abs()

    corr_df_sorted = corr_df.sort_values(by="pearson_r", ascending=False)

    pos = corr_df.sort_values("pearson_r", ascending=False).head(top_k)
    neg = corr_df.sort_values("pearson_r", ascending=True).head(top_k)

    correlated_snps = pd.DataFrame({"names": list(pos["snp"]) + list(neg["snp"])[::-1]})
    return mean_abs_corr, corr_df_sorted, correlated_snps
