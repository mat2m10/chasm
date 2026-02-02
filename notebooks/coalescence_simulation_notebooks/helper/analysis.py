import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def center_geno_by_ps(geno: pd.DataFrame, ps: pd.Series | np.ndarray) -> pd.DataFrame:
    """
    Implements: geno_corr = geno - ((ps - (1-ps)) + 1/2)
    """
    ps = np.asarray(ps, dtype=float)
    offset = (ps - (1 - ps)) + 0.5  # == 2*ps - 0.5
    return geno.astype(float) - offset

def snp_pca_with_gwas_beta(
    geno: pd.DataFrame,
    gwas: pd.DataFrame,
    n_components: int = 2,
    scaler: StandardScaler | None = None,
):
    """
    PCA on SNPs (geno.T): rows are SNPs, columns are samples.

    Returns:
      pca_gwas: DataFrame with PC1..PCk and beta, indexed by SNP name.
      pca_model: fitted PCA
    """
    if scaler is None:
        scaler = StandardScaler(with_mean=True, with_std=True)

    # SNP x sample
    X = geno.T.to_numpy(dtype=float)

    # standardize across samples
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pcs,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=geno.columns,
    )

    # merge GWAS beta
    beta = gwas.set_index("snp")[["beta"]]
    pca_gwas = pca_df.join(beta, how="inner")

    return pca_gwas, pca


def pick_correlated_snps(corr_df_sorted: pd.DataFrame, k: int = 2, exclude_self: bool = True):
    """
    Pick top-k positively correlated and top-k negatively correlated SNP names
    from corr_df_sorted (output of snp_correlation_analysis).

    Assumes corr_df_sorted has columns: ['snp','pearson_r','abs_r'] or at least ['snp','pearson_r'].
    """
    df = corr_df_sorted.copy()

    # Optionally drop the first row if it's the chosen SNP (often pearson_r ~ 1)
    if exclude_self and len(df) > 0:
        df = df.iloc[1:].reset_index(drop=True)

    # Most positive correlations (largest pearson_r)
    pos = df.sort_values("pearson_r", ascending=False).head(k)["snp"].tolist()

    # Most negative correlations (smallest pearson_r)
    neg = df.sort_values("pearson_r", ascending=True).head(k)["snp"].tolist()

    return {"pos": pos, "neg": neg, "all": pos + neg}


def _zscore(x: pd.Series, eps: float = 1e-8) -> pd.Series:
    x = x.astype(float)
    return (x - x.mean()) / (x.std(ddof=0) + eps)


def build_effects_from_correlated_snps(
    geno: pd.DataFrame,
    pheno: pd.DataFrame,
    chosen_snp: str,
    chosen_bias: str,
    corr_df_sorted: pd.DataFrame,
    n_pos: int = 2,
    n_neg: int = 2,
    beta_snp: float = 1.0,
    beta_pop: float = 1.0,
    eps: float = 1e-8,
):
    """
    Replaces your notebook cell that:
      - selects corr SNPs
      - builds effects['snp'], effects['pop'], effects['poly'], effects['pheno']
      - keeps individual correlated components too

    Returns
    -------
    effects : pd.DataFrame
        Columns: snp, pop, poly, pheno, plus corr components like c_pos1, c_neg1, ...
    chosen_corr : dict
        Output of pick_correlated_snps with selected SNP names.
    """
    chosen_corr = pick_correlated_snps(corr_df_sorted, k=max(n_pos, n_neg), exclude_self=True)
    pos = chosen_corr["pos"][:n_pos]
    neg = chosen_corr["neg"][:n_neg]

    effects = pd.DataFrame(index=geno.index)

    # core terms
    effects["snp"] = beta_snp * _zscore(geno[chosen_snp], eps=eps)
    effects["pop"] = beta_pop * _zscore(pheno[chosen_bias], eps=eps)

    # correlated SNP components
    poly_terms = []
    for i, snp in enumerate(pos, start=1):
        col = f"c_pos{i}"
        effects[col] = beta_snp * _zscore(geno[snp], eps=eps)
        poly_terms.append(col)

    for i, snp in enumerate(neg, start=1):
        col = f"c_neg{i}"
        effects[col] = beta_snp * _zscore(geno[snp], eps=eps)
        poly_terms.append(col)

    effects["poly"] = effects[poly_terms].sum(axis=1) if poly_terms else 0.0
    effects["pheno"] = effects["snp"] + effects["pop"] + effects["poly"]

    chosen_corr = {
        "chosen_snp": chosen_snp,
        "chosen_bias": chosen_bias,
        "pos": pos,
        "neg": neg,
        "all": pos + neg,
    }

    return effects, chosen_corr


def summarize_component_fits(geno: pd.DataFrame, effects: pd.DataFrame, chosen_snp: str, chosen_corr: dict):
    """
    Produce the little slope/intercept/-log10(p) table you print in the notebook.

    Returns a DataFrame (instead of printing) so it’s notebook-friendly.
    """
    def fit_line(x_, y_):
        slope, intercept, r, p, se = stats.linregress(x_.astype(float), y_.astype(float))
        neglogp = -np.log10(p) if p > 0 else np.inf
        return slope, intercept, neglogp

    rows = []
    rows.append(("SNP",) + fit_line(geno[chosen_snp], effects["snp"]))
    rows.append(("POP",) + fit_line(geno[chosen_snp], effects["pop"]))
    rows.append(("POLY",) + fit_line(geno[chosen_snp], effects["poly"]))

    # individual correlated components
    for i, snp in enumerate(chosen_corr.get("pos", []), start=1):
        col = f"c_pos{i}"
        rows.append((f"P{i}",) + fit_line(geno[snp], effects[col]))

    for i, snp in enumerate(chosen_corr.get("neg", []), start=1):
        col = f"c_neg{i}"
        rows.append((f"N{i}",) + fit_line(geno[snp], effects[col]))

    out = pd.DataFrame(rows, columns=["Effect", "Slope", "Intercept", "neg_log10p"])
    return out


def gwas_linregress(geno: pd.DataFrame, y: pd.Series | np.ndarray):
    """
    Vectorized-ish GWAS loop (simple linregress per SNP).
    Returns DataFrame: snp, beta, intercept, neglog10p
    """
    y = np.asarray(y, dtype=float)
    cols = list(geno.columns)

    betas = np.empty(len(cols), dtype=float)
    intercepts = np.empty(len(cols), dtype=float)
    neglogps = np.empty(len(cols), dtype=float)

    for j, snp in enumerate(cols):
        x = geno[snp].to_numpy(dtype=float)
        slope, intercept, r, p, se = stats.linregress(x, y)
        betas[j] = slope
        intercepts[j] = intercept
        neglogps[j] = -np.log10(p) if p > 0 else np.inf

    return pd.DataFrame({"snp": cols, "beta": betas, "intercept": intercepts, "neglog10p": neglogps})


def snp_pca_with_gwas_beta(
    geno: pd.DataFrame,
    gwas: pd.DataFrame,
    n_components: int = 2,
    scaler: StandardScaler | None = None,
):
    """
    PCA on SNPs (geno.T): rows are SNPs, columns are samples.

    Returns:
      pca_gwas: DataFrame with PC1..PCk and beta, indexed by SNP name.
      pca_model: fitted PCA
    """
    if scaler is None:
        scaler = StandardScaler(with_mean=True, with_std=True)

    # SNP x sample
    X = geno.T.to_numpy(dtype=float)

    # standardize across samples
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pcs,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=geno.columns,
    )

    # merge GWAS beta
    beta = gwas.set_index("snp")[["beta"]]
    pca_gwas = pca_df.join(beta, how="inner")

    return pca_gwas, pca
