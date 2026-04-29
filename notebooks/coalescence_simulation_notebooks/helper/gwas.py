import pandas as pd
import numpy as np
from scipy import stats
def run_gwas(geno, y, *covariates):
    X = geno.replace([np.inf, -np.inf], np.nan).values

    cov_names = []
    if covariates:
        cov_df = pd.concat(covariates, axis=1)
        C = cov_df.values.astype(float)
        cov_names = list(cov_df.columns)
    else:
        C = None

    n_snps = X.shape[1]
    betas = np.full(n_snps, np.nan)
    pvals = np.full(n_snps, np.nan)
    ns = np.zeros(n_snps, dtype=int)

    # Storage for covariate results: one row per SNP
    cov_betas = {name: np.full(n_snps, np.nan) for name in cov_names}
    cov_pvals = {name: np.full(n_snps, np.nan) for name in cov_names}

    for j in range(n_snps):
        mask = np.isfinite(X[:, j])
        if C is not None:
            mask = mask & np.all(np.isfinite(C), axis=1)

        xj, yj = X[mask, j], y[mask]
        n = mask.sum()
        if n < 10:
            continue

        # Design matrix: intercept | covariates | snp
        if C is not None:
            D = np.column_stack([np.ones(n), C[mask], xj])
        else:
            D = np.column_stack([np.ones(n), xj])

        coefs, _, _, _ = np.linalg.lstsq(D, yj, rcond=None)
        # coefs: [intercept, cov1..., snp]
        snp_coef = coefs[-1]

        # Residuals and SE for all coefficients
        resid = yj - D @ coefs
        rss = resid @ resid
        df = n - D.shape[1]
        if df < 1:
            continue

        var_coefs = rss / df * np.linalg.inv(D.T @ D).diagonal()
        se_coefs = np.sqrt(var_coefs)

        # SNP
        t = snp_coef / se_coefs[-1]
        pvals[j] = 2 * stats.t.sf(np.abs(t), df=df)
        betas[j] = snp_coef
        ns[j] = n

        # Covariates (indices 1..k, skipping intercept at 0)
        for k, name in enumerate(cov_names):
            t_cov = coefs[k + 1] / se_coefs[k + 1]
            cov_betas[name][j] = coefs[k + 1]
            cov_pvals[name][j] = 2 * stats.t.sf(np.abs(t_cov), df=df)

    log10p = -np.log10(np.clip(pvals, 1e-300, 1.0))

    result = pd.DataFrame(
        {"SNP": geno.columns, "beta": betas, "pval": pvals, "logp": log10p, "n": ns}
    )

    for name in cov_names:
        result[f"beta_{name}"] = cov_betas[name]
        result[f"pval_{name}"] = cov_pvals[name]
        result[f"logp_{name}"] = -np.log10(np.clip(cov_pvals[name], 1e-300, 1.0))

    return result
"""
def run_chasm(geno, y, cov1, cov2):
    X = geno.replace([np.inf, -np.inf], np.nan).values
    C1 = cov1.replace([np.inf, -np.inf], np.nan).values
    C2 = cov2.replace([np.inf, -np.inf], np.nan).values

    n_snps = X.shape[1]

    # Storage
    betas = np.full(n_snps, np.nan)
    pvals = np.full(n_snps, np.nan)
    b_cov1 = np.full(n_snps, np.nan)
    p_cov1 = np.full(n_snps, np.nan)
    b_cov2 = np.full(n_snps, np.nan)
    p_cov2 = np.full(n_snps, np.nan)
    ns = np.zeros(n_snps, dtype=int)

    for j in range(n_snps):
        xj = X[:, j]
        c1j = C1[:, j]
        c2j = C2[:, j]

        # Valid rows: finite in all three
        mask = np.isfinite(xj) & np.isfinite(c1j) & np.isfinite(c2j)
        n = mask.sum()
        if n < 10:
            continue

        # Design matrix: intercept | cov1_j | cov2_j | snp_j
        D = np.column_stack([np.ones(n), c1j[mask], c2j[mask], xj[mask]])
        yj = y[mask]

        coefs, _, _, _ = np.linalg.lstsq(D, yj, rcond=None)
        # coefs: [intercept, cov1, cov2, snp]

        resid = yj - D @ coefs
        rss = resid @ resid
        df = n - D.shape[1]
        if df < 1:
            continue

        se = np.sqrt(rss / df * np.linalg.inv(D.T @ D).diagonal())

        def tpval(coef, se, df):
            t = coef / se
            return 2 * stats.t.sf(np.abs(t), df=df)

        betas[j] = coefs[3]
        pvals[j] = tpval(coefs[3], se[3], df)
        b_cov1[j] = coefs[1]
        p_cov1[j] = tpval(coefs[1], se[1], df)
        b_cov2[j] = coefs[2]
        p_cov2[j] = tpval(coefs[2], se[2], df)
        ns[j] = n

    def logp(p):
        return -np.log10(np.clip(p, 1e-300, 1.0))

    return pd.DataFrame(
        {
            "SNP": geno.columns,
            "beta": betas,
            "pval": pvals,
            "logp": logp(pvals),
            "beta_cov1": b_cov1,
            "pval_cov1": p_cov1,
            "logp_cov1": logp(p_cov1),
            "beta_cov2": b_cov2,
            "pval_cov2": p_cov2,
            "logp_cov2": logp(p_cov2),
            "n": ns,
        }
    )
"""

def run_chasm(geno, y, cov1):
    X = geno.replace([np.inf, -np.inf], np.nan).values
    C1 = cov1.replace([np.inf, -np.inf], np.nan).values

    n_snps = X.shape[1]

    # Storage
    betas = np.full(n_snps, np.nan)
    pvals = np.full(n_snps, np.nan)
    b_cov1 = np.full(n_snps, np.nan)
    p_cov1 = np.full(n_snps, np.nan)
    ns = np.zeros(n_snps, dtype=int)

    for j in range(n_snps):
        xj = X[:, j]
        c1j = C1[:, j]

        # Valid rows: finite in all three
        mask = np.isfinite(xj) & np.isfinite(c1j)
        n = mask.sum()
        if n < 10:
            continue

        # Design matrix: intercept | cov1_j | cov2_j | snp_j
        D = np.column_stack([np.ones(n), c1j[mask], xj[mask]])
        yj = y[mask]

        coefs, _, _, _ = np.linalg.lstsq(D, yj, rcond=None)
        # coefs: [intercept, cov1, cov2, snp]

        resid = yj - D @ coefs
        rss = resid @ resid
        df = n - D.shape[1]
        if df < 1:
            continue

        se = np.sqrt(rss / df * np.linalg.inv(D.T @ D).diagonal())

        def tpval(coef, se, df):
            t = coef / se
            return 2 * stats.t.sf(np.abs(t), df=df)

        betas[j] = coefs[2]
        pvals[j] = tpval(coefs[2], se[2], df)
        b_cov1[j] = coefs[1]
        p_cov1[j] = tpval(coefs[1], se[1], df)
        ns[j] = n

    def logp(p):
        return -np.log10(np.clip(p, 1e-300, 1.0))

    return pd.DataFrame(
        {
            "SNP": geno.columns,
            "beta": betas,
            "pval": pvals,
            "logp": logp(pvals),
            "beta_cov1": b_cov1,
            "pval_cov1": p_cov1,
            "logp_cov1": logp(p_cov1),
            "n": ns,
        }
    )
    
def predict_gwas(geno, results):
    X  = geno.replace([np.inf, -np.inf], np.nan).values
    n_snps = X.shape[1]
    b_snp  = results['beta'].values       # (8000,)

    # Replace nan betas with 0 so they contribute nothing
    b_snp  = np.nan_to_num(b_snp)

    # Replace nan/inf in matrices with 0 too
    X  = np.nan_to_num(X)
    return (X @ b_snp)/n_snps
"""
def predict_chasm(geno, cov1, cov2, results):
    geno_filtered = geno[geno.columns[geno.columns.isin(results["SNP"])]]
    cov1_filtered = cov1[cov1.columns[cov1.columns.isin(results["SNP"])]]
    cov2_filtered = cov2[cov2.columns[cov2.columns.isin(results["SNP"])]]
    
    X  = geno_filtered.replace([np.inf, -np.inf], np.nan).values
    n_snps = X.shape[1]
    
    C1 = cov1_filtered.replace([np.inf, -np.inf], np.nan).values
    C2 = cov2_filtered.replace([np.inf, -np.inf], np.nan).values

    b_snp  = results['beta'].values       # (8000,)
    b_cov1 = results['beta_cov1'].values  # (8000,)
    b_cov2 = results['beta_cov2'].values  # (8000,)

    # Replace nan betas with 0 so they contribute nothing
    b_snp  = np.nan_to_num(b_snp)
    b_cov1 = np.nan_to_num(b_cov1)
    b_cov2 = np.nan_to_num(b_cov2)

    # Replace nan/inf in matrices with 0 too
    X  = np.nan_to_num(X)
    C1 = np.nan_to_num(C1)
    C2 = np.nan_to_num(C2)

    # Each is (2000, 8000) dot (8000,) -> (2000,)
    pred = X @ b_snp + C1 @ b_cov1 + C2 @ b_cov2

    return (X @ b_snp)/n_snps, (C1 @ b_cov1 + C2 @ b_cov2)/n_snps
"""
def predict_chasm(geno, cov1, results):
    geno_filtered = geno[geno.columns[geno.columns.isin(results["SNP"])]]
    cov1_filtered = cov1[cov1.columns[cov1.columns.isin(results["SNP"])]]
    
    X  = geno_filtered.replace([np.inf, -np.inf], np.nan).values
    n_snps = X.shape[1]
    
    C1 = cov1_filtered.replace([np.inf, -np.inf], np.nan).values

    b_snp  = results['beta'].values       # (8000,)
    b_cov1 = results['beta_cov1'].values  # (8000,)

    # Replace nan betas with 0 so they contribute nothing
    b_snp  = np.nan_to_num(b_snp)
    b_cov1 = np.nan_to_num(b_cov1)

    # Replace nan/inf in matrices with 0 too
    X  = np.nan_to_num(X)
    C1 = np.nan_to_num(C1)

    # Each is (2000, 8000) dot (8000,) -> (2000,)
    pred = X @ b_snp + C1 @ b_cov1

    return (X @ b_snp)/n_snps, (C1 @ b_cov1)/n_snps

