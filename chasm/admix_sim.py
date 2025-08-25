
"""
admix_sim.py — minimal, fast simulator for admixed genotype matrices.

Model
-----
- Global ancestral allele frequency p ~ Beta(a, b) (skewed toward rare by default).
- K source populations with drift Fst via Balding–Nichols:
      p_k ~ Beta(p*(1-Fst)/Fst, (1-p)*(1-Fst)/Fst), independently per SNP.
- For each individual i with ancestry proportions w_i (length K, sum to 1):
    * For each of the 2 chromosome copies, draw an ancestral pop Z ~ Categorical(w_i).
    * Allele for that copy is Bernoulli(p_{Z}). Genotype is the sum over two draws.
  This approximates local ancestry switching without explicit tracts.

API
---
- sample_ancestry_proportions(n_ind, K, centers=None, conc=100.)
    Returns n_ind x K matrix of Dirichlet draws around "centers".
- simulate_admixed_genotypes(n_ind, n_snps, K=2, Fst=0.05, a=0.4, b=4.0,
                             W=None, seed=None, return_pops=False)
    Returns (G, W, P) if return_pops else G,
    where G is n_ind x n_snps diploid genotypes in {0,1,2},
          W is n_ind x K ancestry proportions,
          P is dictionary with 'p_anc' and 'p_src' (K x n_snps).
"""

from __future__ import annotations

import numpy as np

def _rng(seed):
    return np.random.default_rng(seed)

def sample_ancestry_proportions(n_ind: int, K: int, centers: np.ndarray | None = None,
                                conc: float = 100.0, seed: int | None = None) -> np.ndarray:
    """
    Draw individual ancestry proportions W (n_ind x K).
    If `centers` is None: use uniform Dirichlet(1,...,1).
    Else: for each individual, sample Dirichlet(conc * centers[k*]), where centers
    is either (K,) or (n_ind, K). Larger `conc` => tighter around centers.
    """
    rng = _rng(seed)
    if centers is None:
        alpha = np.ones(K)
        return rng.dirichlet(alpha, size=n_ind)
    centers = np.asarray(centers, dtype=float)
    if centers.ndim == 1:
        alpha = conc * (centers / centers.sum())
        return rng.dirichlet(alpha, size=n_ind)
    if centers.shape != (n_ind, K):
        raise ValueError("centers must be shape (K,) or (n_ind, K)")
    out = np.empty_like(centers)
    for i in range(n_ind):
        alpha = conc * (centers[i] / centers[i].sum())
        out[i] = rng.dirichlet(alpha)
    return out

def _balding_nichols(p: np.ndarray, Fst: float, K: int, rng) -> np.ndarray:
    """
    Given global freq vector p (n_snps,), draw K population-specific frequencies.
    Returns array (K, n_snps).
    """
    if Fst <= 0 or Fst >= 1:
        raise ValueError("Fst must be in (0,1)")
    p = np.clip(p, 1e-6, 1 - 1e-6)
    a = p * (1 - Fst) / Fst
    b = (1 - p) * (1 - Fst) / Fst
    P = np.empty((K, p.size), dtype=float)
    for k in range(K):
        P[k] = rng.beta(a, b)
    return np.clip(P, 1e-6, 1 - 1e-6)

def simulate_admixed_genotypes(n_ind: int, n_snps: int, K: int = 2, Fst: float = 0.05,
                               a: float = 0.4, b: float = 4.0,
                               W: np.ndarray | None = None,
                               seed: int | None = None,
                               return_pops: bool = False):
    """
    Simulate diploid genotypes under a simple admixture with K sources.
    Parameters
    ----------
    n_ind : number of individuals
    n_snps: number of SNPs
    K     : number of sources
    Fst   : Balding–Nichols drift parameter
    a,b   : Beta(a,b) hyperparameters for global p (default skews rare variants)
    W     : optional n_ind x K ancestry matrix; if None, Dirichlet draws used
    seed  : RNG seed
    return_pops : if True, also return (W, pop-frequency dictionary)

    Returns
    -------
    G or (G, W, P): G is (n_ind, n_snps) with values in {0,1,2}
    """
    rng = _rng(seed)

    if W is None:
        W = rng.dirichlet(alpha=np.ones(K), size=n_ind)
    else:
        W = np.asarray(W, dtype=float)
        if W.shape != (n_ind, K):
            raise ValueError("W must have shape (n_ind, K)")
        # ensure rows sum to 1
        W = (W.T / W.sum(axis=1)).T

    # global allele frequencies (skewed toward rare)
    p_anc = rng.beta(a, b, size=n_snps)
    # source-pop frequencies via BN model
    p_src = _balding_nichols(p_anc, Fst, K, rng)  # shape (K, n_snps)

    # For each individual & SNP, draw 2 ancestry labels and alleles
    G = np.zeros((n_ind, n_snps), dtype=np.uint8)

    # Vectorised approach: for each ind, for each copy (2), draw ancestral pop and allele
    for i in range(n_ind):
        # ancestry labels for two haplotypes
        z1 = rng.choice(K, size=n_snps, p=W[i])
        z2 = rng.choice(K, size=n_snps, p=W[i])
        # alleles
        a1 = rng.binomial(1, p_src[z1, np.arange(n_snps)])
        a2 = rng.binomial(1, p_src[z2, np.arange(n_snps)])
        G[i] = a1 + a2

    if return_pops:
        return G, W, {'p_anc': p_anc, 'p_src': p_src}
    return G
