import numpy as np
import pandas as pd

def simulate_inbred(df_binary: pd.DataFrame, F: float, seed: int = 123, do_random_flip: bool = True):
    """
    Replicates the R-style pipeline you posted:

    1) For each population and each SNP, optionally flip ONE random individual's 0/1.
    2) Estimate q (minor-allele frequency) within each population from the (possibly flipped) 0/1 matrix.
    3) Simulate diploid genotypes per SNP under HWE with inbreeding F using probs:
           [-1, 0, 1] ~ [q^2 + F*p*q,  2*p*q*(1-F),  p^2 + F*p*q]
       (coding: -1 = minor hom, 0 = hetero, 1 = major hom)
    4) Across all individuals, flip alleles per SNP so that MAF ≤ 0.5.
    5) Sort SNPs by descending MAF and rename columns to 'i_MAF_{maf:.3f}'.

    Returns
    -------
    complete : pd.DataFrame
        Genotype matrix encoded as -1/0/1 with a 'populations' column.
    maf_series : pd.Series
        Final MAF per SNP (after possible allele flipping), indexed by new SNP names.
    """
    rng = np.random.default_rng(seed)

    # Identify SNP columns and populations
    Vcols = [c for c in df_binary.columns if c.startswith("V")]
    pops = df_binary["populations"].to_numpy()
    unique_pops = np.unique(pops)

    # Work on a copy
    work = df_binary.copy()

    # 1) Optional: flip one random genotype per SNP within each population
    if do_random_flip:
        for pop in unique_pops:
            mask = pops == pop
            # indices in the full dataframe for this pop
            idx_pop = np.flatnonzero(mask)
            # for each SNP, choose one row index from this pop to flip
            # vectorized-ish: loop over columns (fast enough for large L; avoids Python per-row loops)
            for snp in Vcols:
                i = int(rng.integers(0, idx_pop.size))
                ridx = idx_pop[i]
                work.at[ridx, snp] = 1 - int(work.at[ridx, snp])

    # 2) Estimate q within each population and simulate genotypes under HWE+F
    N = len(work)
    S = len(Vcols)
    G_out = np.zeros((N, S), dtype=np.int8)  # -1/0/1

    for pop in unique_pops:
        mask = pops == pop
        X = work.loc[mask, Vcols].to_numpy(dtype=float)  # 0/1 presence matrix
        q = X.mean(axis=0)
        p = 1.0 - q
        P_minus1 = q**2 + F * p * q            # minor hom (coded -1)
        P_0      = 2.0 * p * q * (1.0 - F)     # heterozygote (0)
        # P_plus1 = p**2 + F*p*q               # major hom (+1) = 1 - (P_minus1 + P_0)

        # sample for all individuals × SNPs at once using uniforms
        U = rng.random((mask.sum(), S))
        T0 = P_minus1
        T1 = P_minus1 + P_0

        # Assign -1 if U < T0; 0 if T0 <= U < T1; 1 otherwise
        block = np.ones_like(U, dtype=np.int8)
        block[U < T1] = 0
        block[U < T0] = -1
        G_out[mask] = block

    complete = pd.DataFrame(G_out, columns=Vcols, index=work.index)
    complete["populations"] = work["populations"].values

    # 3) Flip alleles per SNP so MAF ≤ 0.5 and compute MAF
    # With -1/0/1 coding: minor allele count per person is (genotype == -1)*2 + (genotype == 0)*1
    N = len(complete)
    maf_values = {}
    for snp in Vcols:
        g = complete[snp].to_numpy()
        # counts:
        count_mh = np.count_nonzero(g == -1)   # minor hom
        count_het = np.count_nonzero(g == 0)   # hetero
        maf = (2 * count_mh + 1 * count_het) / (2.0 * N)
        if maf > 0.5:
            # flip allele labels: -1 <-> 1 (hetero stays 0)
            g_flipped = g.copy()
            g_flipped[g == -1] = 1
            g_flipped[g == 1] = -1
            complete[snp] = g_flipped
            maf = 1.0 - maf
        maf_values[snp] = float(maf)

    # 4) Sort SNPs by descending MAF and rename as V1_MAF_{...}, V2_MAF_{...}, ...
    sorted_snps = sorted(maf_values.items(), key=lambda x: x[1], reverse=True)
    new_names = {old: f"V{i+1}_MAF_{maf:.3f}" for i, (old, maf) in enumerate(sorted_snps)}
    complete.rename(columns=new_names, inplace=True)

    ordered_cols = [new_names[v] for v, _ in sorted_snps] + ["populations"]
    complete = complete[ordered_cols]

    maf_series = pd.Series({new_names[k]: v for k, v in maf_values.items()})
    return complete, maf_series
