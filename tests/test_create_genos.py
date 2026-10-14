import numpy as np
import pandas as pd
import pytest

from scripts.create_genos import categorize_by_maf, compute_maf, format_M, simulate_genotypes


def test_compute_maf_known_values():
    # 4 diploid individuals, 0/1/2 minor-allele counts
    snps = pd.DataFrame(
        {
            "all_ref": [0, 0, 0, 0],   # af 0      -> maf 0
            "quarter": [1, 1, 0, 0],   # af 2/8    -> maf 0.25
            "half":    [2, 2, 0, 0],   # af 4/8    -> maf 0.5
            "flipped": [2, 2, 2, 0],   # af 6/8    -> maf 0.25 (folded)
        }
    )
    maf = compute_maf(snps)
    assert maf.name == "MAF"
    assert list(maf.index) == list(snps.columns)
    np.testing.assert_allclose(maf.to_numpy(), [0.0, 0.25, 0.5, 0.25])


def test_compute_maf_is_folded():
    rng = np.random.default_rng(0)
    snps = pd.DataFrame(rng.integers(0, 3, size=(50, 20)))
    maf = compute_maf(snps)
    assert ((maf >= 0) & (maf <= 0.5)).all()


def test_categorize_by_maf_thresholds():
    maf = pd.Series([0.0, 0.004, 0.005, 0.0051, 0.01, 0.2, 0.5])
    cats = categorize_by_maf(maf).astype(str).tolist()
    assert cats == ["very_rare", "very_rare", "very_rare", "rare", "rare", "common", "common"]


@pytest.mark.parametrize("M, expected", [(1.0, "1"), (0.1, "0.1"), (2.5, "2.5")])
def test_format_M(M, expected):
    assert format_M(M) == expected


def _haploid_input(n_pops=2, n_per_pop=300, n_snps=40, seed=1):
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.1, 0.9, size=n_snps)
    blocks = []
    for pop in range(1, n_pops + 1):
        block = pd.DataFrame(
            (rng.random((n_per_pop, n_snps)) < freqs).astype(int),
            columns=[f"snp{i}" for i in range(n_snps)],
        )
        block["populations"] = pop
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def test_simulate_genotypes_shape_coding_and_order():
    np.random.seed(123)
    snps = _haploid_input()
    geno, maf = simulate_genotypes(snps, F=0.0)

    n_snps = snps.shape[1] - 1
    assert geno.shape == (len(snps), n_snps)
    assert "populations" not in geno.columns
    assert set(np.unique(geno.to_numpy())) <= {-1, 0, 1}

    # MAF folded; columns renamed "<rank>_MAF_<maf>" and sorted by descending MAF.
    # Note: the returned maf Series is label-aligned with geno, not positionally.
    assert ((maf >= 0) & (maf <= 0.5)).all()
    assert set(maf.index) == set(geno.columns)
    maf = maf[geno.columns]
    assert maf.is_monotonic_decreasing
    assert geno.columns[0].startswith("1_MAF_")

    # Reported MAF matches the genotypes (-1 = minor homozygote)
    G = geno.to_numpy()
    recomputed = ((G == -1).sum(0) * 2 + (G == 0).sum(0)) / (2.0 * len(G))
    np.testing.assert_allclose(recomputed, maf.to_numpy())


def test_simulate_genotypes_hardy_weinberg_when_outbred():
    np.random.seed(7)
    snps = _haploid_input(n_pops=1, n_per_pop=4000, n_snps=10, seed=3)
    geno, maf = simulate_genotypes(snps, F=0.0)
    maf = maf[geno.columns]
    G = geno.to_numpy()
    het_obs = (G == 0).mean(axis=0)
    het_exp = 2 * maf.to_numpy() * (1 - maf.to_numpy())
    np.testing.assert_allclose(het_obs, het_exp, atol=0.03)


def test_simulate_genotypes_full_inbreeding_has_no_heterozygotes():
    np.random.seed(11)
    snps = _haploid_input(n_pops=1, n_per_pop=500, n_snps=10)
    geno, _ = simulate_genotypes(snps, F=1.0)
    assert not (geno.to_numpy() == 0).any()
