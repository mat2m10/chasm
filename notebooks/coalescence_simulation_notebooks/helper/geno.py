import pandas as pd

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

import numpy as np
import pandas as pd

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