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
