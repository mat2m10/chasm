import os
import pandas as pd
import numpy as np

def _get_snp_cols(geno: pd.DataFrame):
    exclude = {"x", "y", "populations", "population", "pop", "iid", "id"}
    numeric_cols = [c for c in geno.columns if pd.api.types.is_numeric_dtype(geno[c])]
    snp_cols = [c for c in numeric_cols if c not in exclude]
    if len(snp_cols) == 0:
        raise ValueError("No SNP columns found. Ensure geno has numeric SNP columns.")
    return snp_cols


def polygenic_noise(
    geno: pd.DataFrame,
    chosen_snp: str,
    p_causal: float = 0.01,
    n_causal: int | None = None,
    total_beta: float = 0.2,
    seed: int = 1,
    downweight_by_maf_sd: bool = True,
    regen: bool = False,
    cache_path: str = "g_noise.npz",
):
    # --- cache ---
    if (not regen) and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return (
            d["g_std"],
            d["true_beta"],
            float(d["true_intercept"]),
            d["true_beta_std"],
            float(d["true_intercept_std"]),
            d["is_causal"].astype(bool),
            list(d["cols"]),
        )

    cols = _get_snp_cols(geno)
    if chosen_snp not in cols:
        raise ValueError(f"{chosen_snp} not in geno SNP columns")

    G = geno[cols].to_numpy(dtype=np.float32)
    N, M = G.shape

    j = cols.index(chosen_snp)
    rng = np.random.default_rng(seed)

    eligible = np.arange(M, dtype=int)
    eligible = eligible[eligible != j]

    # choose causal SNPs (excluding chosen_snp)
    if n_causal is not None:
        n_causal = int(max(0, min(int(n_causal), eligible.size)))
        causal_idx = rng.choice(eligible, size=n_causal, replace=False)
    else:
        p_causal = float(max(0.0, min(float(p_causal), 1.0)))
        n_causal_eff = int(np.round(p_causal * eligible.size))
        causal_idx = rng.choice(eligible, size=n_causal_eff, replace=False)

    is_causal = np.zeros(M, dtype=bool)
    is_causal[causal_idx] = True

    # raw beta draw
    true_beta = np.zeros(M, dtype=np.float32)
    true_beta[is_causal] = rng.normal(0.0, 1.0, size=is_causal.sum()).astype(np.float32)

    # optional downweight by MAF sd
    if downweight_by_maf_sd:
        maf = np.array([float(c.split("_MAF_")[-1]) for c in cols], dtype=np.float32)
        sd = np.sqrt(2.0 * maf * (1.0 - maf))
        sd[sd == 0] = 1.0
        true_beta = true_beta / sd

    # force chosen_snp non-causal
    true_beta[j] = 0.0

    # scale beta vector to have L2 norm = total_beta
    norm = float(np.linalg.norm(true_beta))
    if norm > 0:
        true_beta *= (float(total_beta) / norm)

    # ---- define raw model with explicit intercept ----
    # g_raw = G @ true_beta + true_intercept
    g_raw = G @ true_beta
    true_intercept = 0.0  # keep it explicit (you can change later if you want)

    # ---- standardize and convert to equivalent (beta_std, intercept_std) ----
    mu = float((g_raw + true_intercept).mean())
    sigma = float((g_raw + true_intercept).std() + 1e-8)

    # g_std = (g_raw + true_intercept - mu) / sigma
    #      = G @ (true_beta/sigma) + (true_intercept - mu)/sigma
    true_beta_std = true_beta / sigma
    true_intercept_std = (true_intercept - mu) / sigma
    g_std = (g_raw + true_intercept - mu) / sigma

    # cache
    np.savez_compressed(
        cache_path,
        g_std=g_std.astype(np.float32),
        true_beta=true_beta.astype(np.float32),
        true_intercept=np.array(true_intercept, dtype=np.float32),
        true_beta_std=true_beta_std.astype(np.float32),
        true_intercept_std=np.array(true_intercept_std, dtype=np.float32),
        is_causal=is_causal.astype(np.uint8),
        cols=np.array(cols, dtype=object),
        chosen_snp=np.array(chosen_snp),
        seed=np.array(seed),
        p_causal=np.array(p_causal),
        n_causal=np.array(-1 if n_causal is None else int(n_causal)),
        total_beta=np.array(total_beta),
        downweight_by_maf_sd=np.array(int(downweight_by_maf_sd)),
    )

    params_df = pd.DataFrame({
        "name_snps": cols,
        "true_beta": true_beta,
        "true_beta_std": true_beta_std,
    })

    # intercepts are scalars → attach as attributes (cleanest)
    params_df.attrs["true_intercept"] = true_intercept
    params_df.attrs["true_intercept_std"] = true_intercept_std

    return g_std, params_df, is_causal
