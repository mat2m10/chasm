"""PC adjustment removes spurious association caused by population structure."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scripts.lower_dim import compute_pcs


def _structured_data(seed=0, n_per_pop=300, n_snps=200):
    rng = np.random.default_rng(seed)
    p_a = rng.uniform(0.05, 0.5, n_snps)
    p_b = np.clip(p_a + rng.normal(0, 0.25, n_snps), 0.02, 0.98)  # strong differentiation
    geno_a = rng.binomial(2, p_a, size=(n_per_pop, n_snps))
    geno_b = rng.binomial(2, p_b, size=(n_per_pop, n_snps))
    geno = pd.DataFrame(np.vstack([geno_a, geno_b]), columns=[f"s{i}" for i in range(n_snps)])
    pop = np.r_[np.zeros(n_per_pop), np.ones(n_per_pop)]
    # Phenotype: environmental difference between populations only, no SNP effect
    y = 2.0 * pop + rng.normal(0, 1, len(pop))
    diff = np.abs(p_a - p_b)
    return geno, y, int(np.argmax(diff))


def test_compute_pcs_shape_and_labels():
    geno, _, _ = _structured_data(n_snps=30)
    pcs = compute_pcs(geno, min_components=4, max_components=10)
    assert pcs.shape == (len(geno), 10)
    assert list(pcs.columns) == [f"PC{i}" for i in range(1, 11)]
    assert pcs.index.equals(geno.index)


def test_pc_adjustment_removes_stratification():
    geno, y, snp_idx = _structured_data()
    x = geno.iloc[:, snp_idx].to_numpy(dtype=float)

    naive = sm.OLS(y, sm.add_constant(x)).fit()
    assert naive.pvalues[1] < 1e-8  # spurious hit driven by ancestry

    pcs = compute_pcs(geno, min_components=4, max_components=4)
    X = sm.add_constant(np.column_stack([x, pcs[["PC1"]].to_numpy()]))
    adjusted = sm.OLS(y, X).fit()
    assert adjusted.pvalues[1] > 1e-3
