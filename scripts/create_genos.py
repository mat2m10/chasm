from typing import Dict, Tuple
import pandas as pd
import numpy as np
import subprocess
def format_M(M):
    return str(int(M)) if M.is_integer() else str(M)

def read_simulated_genotypes(cfg: Dict[str, int]) -> pd.DataFrame:
    """Read the simulated genotype CSV produced by the R script, then delete it.
    Returns a DataFrame with genotype columns only; population labels are added later.
    """
    csv_name = R_FILE_STEM.format(**cfg)
    csv_path = R_DIRECTORY / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Clean up the file after load (optional)
    try:
        csv_path.unlink(missing_ok=True)
    except Exception:
        pass
    return df

def compute_maf(snps: pd.DataFrame) -> pd.Series:
    """Minor allele frequency per SNP assuming 0/1/2 encoding and diploidy."""
    G = snps.to_numpy()
    n_individuals = G.shape[0]
    allele_sum = G.sum(axis=0)  # total minor alleles across individuals
    af = allele_sum / (2.0 * n_individuals)
    maf = np.minimum(af, 1.0 - af)
    return pd.Series(maf, index=snps.columns, name="MAF")

def categorize_by_maf(maf: pd.Series) -> pd.Series:
    """Categorize SNPs into very_rare, rare, common by MAF thresholds."""
    bins = [-1e-9, 0.005, 0.01, 1.0]
    labels = ["very_rare", "rare", "common"]
    return pd.cut(maf, bins=bins, labels=labels, include_lowest=True)

def run_r_generation(G, L, c, k, M, R_SCRIPT, R_DIRECTORY):
    r_expr = (
        f"G <- {G}; L <- {L}; c <- {c}; k <- {k}; M <- {format_M(M)};"
        f"source('{R_SCRIPT}', echo=TRUE)"
    )
    res = subprocess.run(
        ["Rscript", "-e", r_expr],
        capture_output=True,
        text=True,
        cwd=str(R_DIRECTORY),   # << safer than setwd() inside R
    )
    if res.returncode != 0:
        raise RuntimeError(f"R failed.\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}")
    return res

def read_generated_csv(G, L, c, k, M, R_DIRECTORY, CSV_STEM):
    csv_path = R_DIRECTORY / CSV_STEM.format(G=G, L=L, c=c, k=k, M=format_M(M))
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def simulate_genotypes(snps: pd.DataFrame, F: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Simulate genotypes under an inbreeding model per population, 
    flip alleles so MAF <= 0.5, and sort SNPs by descending MAF.

    Parameters
    ----------
    snps : pd.DataFrame
        Input genotype dataframe with SNP columns (coded as 0/1) 
        and one final column 'populations' labeling individuals.
    F : float, default=0.0
        Inbreeding coefficient.

    Returns
    -------
    complete : pd.DataFrame
        Genotype dataframe with simulated genotypes, flipped alleles 
        where MAF > 0.5, and SNPs renamed/sorted by descending MAF. 
        Genotypes encoded as -1, 0, 1 for (aa, aA, AA).
    maf : pd.Series
        Minor allele frequencies (MAF) indexed by the new SNP column names.
    """
    complete = []
    snp_cols = snps.columns[:-1]  # all SNPs; last column is 'populations'

    for pop, pop_df in snps.groupby('populations', sort=False):
        df = pop_df.copy()
        arr = df[snp_cols].to_numpy()                    # shape: (n_individuals, n_snps)
        n_ind, n_snps = arr.shape

        # --- Flip one random genotype per SNP (0<->1), vectorized
        flip_rows = np.random.randint(0, n_ind, size=n_snps)
        arr[flip_rows, np.arange(n_snps)] = 1 - arr[flip_rows, np.arange(n_snps)]

        # --- Estimate MAF q for each SNP
        q = arr.mean(axis=0)                             # shape: (n_snps,)
        p = 1.0 - q

        # --- Build genotype probabilities per SNP under inbreeding F
        # order: [-1, 0, 1] corresponds to (aa, aA, AA)
        probs = np.empty((n_snps, 3), dtype=float)
        probs[:, 0] = q**2 + F*p*q                       # P(aa)   -> -1
        probs[:, 1] = 2*p*q*(1.0 - F)                    # P(aA)   ->  0
        probs[:, 2] = p**2 + F*p*q                       # P(AA)   ->  1
        probs /= probs.sum(axis=1, keepdims=True)        # guard against tiny rounding error

        # --- Sample all genotypes at once using inverse-CDF
        U = np.random.rand(n_ind, n_snps)                # uniforms
        cum = np.cumsum(probs, axis=1)                   # (n_snps, 3)
        cats = (U[..., None] >= cum.reshape(1, n_snps, 3)).sum(axis=2)  # 0,1,2
        geno = np.take(np.array([-1, 0, 1], dtype=int), cats)           # map categories to genotypes

        # --- Store back
        df.loc[:, snp_cols] = geno
        complete.append(df)

    # Combine all populations
    complete = pd.concat(complete, ignore_index=True)
    N = len(complete)

    # --- Flip alleles so MAF <= 0.5 and compute MAFs (vectorized)
    G = complete[snp_cols].to_numpy()
    count_neg1 = (G == -1).sum(axis=0)
    count_zero  = (G == 0).sum(axis=0)
    maf = (2*count_neg1 + count_zero) / (2.0 * N)       # Series aligned to snp_cols
    maf = pd.Series(maf, index=snp_cols)

    # Flip columns with MAF > 0.5 by multiplying by -1 (heterozygotes stay 0)
    to_flip = maf[maf > 0.5].index
    if len(to_flip):
        complete.loc[:, to_flip] = -complete[to_flip]
        maf.loc[to_flip] = 1.0 - maf.loc[to_flip]

    # --- Sort SNPs by descending MAF and rename
    order = maf.sort_values(ascending=False).index
    new_names = {s: f"{i+1}_MAF_{maf[s]:.3f}" for i, s in enumerate(order)}

    complete = complete.rename(columns=new_names)
    complete = complete[list(new_names.values()) + ['populations']]
    complete = complete.drop(columns=['populations'])
    maf = maf.rename(index=new_names)

    return complete, maf
