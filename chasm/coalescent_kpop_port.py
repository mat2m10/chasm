"""
Python ports of the key R/C functions you used for creating admixted genotypes via the (structured) coalescent.

This file re-implements, in pure Python + NumPy:
  • The structured coalescent on a k×k lattice with nearest-neighbour migration (a line-for-line port of the logic in `kpop.c`).
  • Utilities to drop mutations on the tree and build 0/1 haplotype/genotype matrices like your R pipeline produced.

API overview (mirrors the original intent):
  - simulate_tree_kpop_lattice(k, n_per_pop, M, seed=None)
      → dict with 'n_total', 'time', 'D1', 'D2', 'branch_length', 'parent'
  - one_mutation_from_tree(tree, rng=None)
      → 0/1 vector (length n_total) for one segregating site
  - simulate_genotype_matrix_from_tree(tree, n_snps, seed=None)
      → (n_total × n_snps) int array of 0/1 genotypes
  - simulate_kpop_binary_genotypes(k, c, L, M, seed=None, return_pop=True)
      → convenience wrapper that returns a pandas DataFrame like your R CSV (V1..VL plus 'populations')

Notes
-----
• Indices are 0-based (Pythonic). Leaves are nodes 0..(n_total-1). The root has parent = -1. For leaves, D1=D2=-1.
• Random draws follow the same distributions and event logic as `kpop.c`, so results are stochastic but statistically consistent.
• If you later want to match R outputs exactly for a fixed seed, you may still see tiny differences because C's `rand()` and NumPy RNG differ.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class KPopTree:
    n_total: int
    time: np.ndarray           # shape (n_nodes,)
    D1: np.ndarray             # shape (n_nodes,); child index or -1
    D2: np.ndarray             # shape (n_nodes,)
    branch_length: np.ndarray  # shape (n_nodes,); length to parent
    parent: np.ndarray         # shape (n_nodes,); parent index or -1

    @property
    def n_nodes(self) -> int:
        return 2 * self.n_total - 1


# ------------------------------
# Weighted sampling helpers
# ------------------------------

def _sample_weighted(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Return index i with probability proportional to probs[i]."""
    total = probs.sum()
    if total <= 0:
        # If all zero, sample uniformly
        return int(rng.integers(0, len(probs)))
    u = rng.random() * total
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c:
            return i
    return len(probs) - 1


def _sample_population_index(pop_list: np.ndarray, pop: int, n_sum: int, rng: np.random.Generator) -> int:
    """Uniformly sample an *active* lineage index in [0, n_sum) whose population == pop.
    Equivalent to kpop.c's sample_populations()."""
    # Find active indices in the first n_sum entries
    idx = np.flatnonzero(pop_list[:n_sum] == pop)
    if idx.size == 0:
        # Should not happen if called correctly; fallback uniform
        return int(rng.integers(0, n_sum))
    j = int(rng.integers(0, idx.size))
    return int(idx[j])


# ------------------------------
# Structured coalescent on a k×k lattice
# ------------------------------

def simulate_tree_kpop_lattice(
    k: int,
    n_per_pop: int | np.ndarray,
    M: float,
    seed: Optional[int] = None,
) -> KPopTree:
    """Simulate a coalescent tree on a k×k lattice with nearest-neighbour migration.

    Parameters
    ----------
    k : int
        Lattice side length (k×k demes).
    n_per_pop : int | array-like (length k*k)
        Number of sampled lineages per population (deme). If an int, the same
        count is used for every population.
    M : float
        Migration parameter (matches `kpop.c` interpretation).
    seed : int, optional
        RNG seed.

    Returns
    -------
    KPopTree
        Tree with times, topology (D1/D2), parent links and branch lengths.
    """
    rng = np.random.default_rng(seed)

    if isinstance(n_per_pop, int):
        n_vec = np.full(k * k, int(n_per_pop), dtype=int)
    else:
        n_vec = np.array(n_per_pop, dtype=int).copy()
        assert n_vec.size == k * k, "n_per_pop must have length k*k"

    if M <= 0:
        # Mimic kpop.c behaviour (forces a small positive M)
        M = 0.01

    n_total = int(n_vec.sum())
    n_nodes = 2 * n_total - 1

    # Outputs
    time = np.zeros(n_nodes, dtype=float)
    D1 = np.full(n_nodes, -1, dtype=int)
    D2 = np.full(n_nodes, -1, dtype=int)
    parent = np.full(n_nodes, -1, dtype=int)
    branch_length = np.zeros(n_nodes, dtype=float)

    # Active lineages (first n_sum entries are live)
    k_list = np.arange(n_total, dtype=int)

    # Population label per *active* lineage position
    pop_list = np.empty(n_total, dtype=int)
    m = 0
    for pop_id in range(k * k):
        cnt = n_vec[pop_id]
        pop_list[m : m + cnt] = pop_id
        m += cnt

    # Build neighbour structure (up to 4-neighbours)
    mig_mat = -np.ones((k * k, 4), dtype=int)
    mig_prob = np.zeros((k * k, 4), dtype=float)

    def _idx(i: int, j: int) -> int:
        return i * k + j

    for i in range(k):
        for j in range(k):
            p = _idx(i, j)
            options: list[Tuple[int, float]] = []
            if i > 0:
                options.append((_idx(i - 1, j), 1.0))
            if i < k - 1:
                options.append((_idx(i + 1, j), 1.0))
            if j > 0:
                options.append((_idx(i, j - 1), 1.0))
            if j < k - 1:
                options.append((_idx(i, j + 1), 1.0))
            # Fill row
            for a, (nb, prob) in enumerate(options):
                mig_mat[p, a] = nb
                mig_prob[p, a] = prob
            if options:
                mig_prob[p, : len(options)] /= float(len(options))

    # Per-population neighbour counts
    mig_rates = (mig_mat >= 0).sum(axis=1)

    t = 0.0
    n_sum = n_total
    current_node = n_total  # next internal node index

    while n_sum > 1:
        # Rates
        rate_co = n_vec * (n_vec - 1.0) / 2.0
        rate_co_tot = float(rate_co.sum())
        rate_mig = (M / 2.0) * n_vec * mig_rates
        rate_mig_tot = float(rate_mig.sum())

        # Time to next event
        lam = rate_co_tot + rate_mig_tot
        if lam <= 0:
            # Shouldn’t happen unless all demes have <=1 lineage and M==0
            break
        t_event = -math.log(rng.random()) / lam
        t += t_event

        # Migration event?
        if rng.random() <= (rate_mig_tot / lam):
            pop_for_migrant = _sample_weighted(rate_mig.astype(float), rng)
            # Choose which active lineage migrates
            idx_lineage = _sample_population_index(pop_list, pop_for_migrant, n_sum, rng)

            # Choose destination among neighbours
            row_probs = mig_prob[pop_for_migrant]
            row_dsts = mig_mat[pop_for_migrant]
            # Mask invalid
            valid = row_dsts >= 0
            dst = row_dsts[valid][_sample_weighted(row_probs[valid], rng)]

            # Update counts and labels
            n_vec[pop_for_migrant] -= 1
            n_vec[dst] += 1
            pop_list[idx_lineage] = int(dst)
        else:
            # Coalescent event in some population
            which_pop = _sample_weighted(rate_co.astype(float), rng)

            a = _sample_population_index(pop_list, which_pop, n_sum, rng)
            b = _sample_population_index(pop_list, which_pop, n_sum, rng)
            while b == a:
                b = _sample_population_index(pop_list, which_pop, n_sum, rng)

            # Children are the nodes referenced by active positions a and b
            child1 = int(k_list[a])
            child2 = int(k_list[b])

            # Set parent time and links
            time[current_node] = t
            D1[current_node] = child1
            D2[current_node] = child2
            parent[child1] = current_node
            parent[child2] = current_node
            branch_length[child1] = t - time[child1]
            branch_length[child2] = t - time[child2]

            # Replace the two active entries by the new node (like kpop.c)
            lo, hi = (a, b) if a < b else (b, a)
            k_list[lo] = current_node
            k_list[hi] = k_list[n_sum - 1]
            pop_list[hi] = pop_list[n_sum - 1]

            # Book-keeping
            n_vec[which_pop] -= 1
            n_sum -= 1
            current_node += 1

    # Root branch_length remains 0 by convention (no parent)
    return KPopTree(
        n_total=n_total,
        time=time,
        D1=D1,
        D2=D2,
        branch_length=branch_length,
        parent=parent,
    )


# ------------------------------
# Drop mutations on the tree to build genotypes
# ------------------------------

def _fill_in_genotype_recursive(n_leaves: int, gt: np.ndarray, D1: np.ndarray, D2: np.ndarray, node: int) -> None:
    """Mark all descendant leaves of `node` with 1 (carrying the mutation)."""
    d1 = D1[node]
    d2 = D2[node]
    if d1 == -1 and d2 == -1:
        # Leaf
        gt[node] = 1
        return
    if d1 < n_leaves:
        gt[d1] = 1
    else:
        _fill_in_genotype_recursive(n_leaves, gt, D1, D2, d1)
    if d2 < n_leaves:
        gt[d2] = 1
    else:
        _fill_in_genotype_recursive(n_leaves, gt, D1, D2, d2)


def one_mutation_from_tree(tree: KPopTree, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Throw a single mutation onto the tree, choosing a branch with probability
    proportional to its length; return a 0/1 vector over leaves.

    Mirrors `one_mutation_from_tree()` + `fill_in_genotypes()` in kpop.c.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_leaves = tree.n_total
    n_nodes = tree.n_nodes

    # Sample a node proportional to branch length
    bl = tree.branch_length.copy()
    # Guard against all-zero (degenerate) lengths
    if bl.sum() <= 0:
        # Fallback: choose any non-root node uniformly
        candidates = np.arange(n_nodes)
        chosen = int(rng.integers(0, n_nodes))
    else:
        chosen = _sample_weighted(bl, rng)

    gt = np.zeros(n_leaves, dtype=int)
    _fill_in_genotype_recursive(n_leaves, gt, tree.D1, tree.D2, chosen)
    return gt


def simulate_genotype_matrix_from_tree(tree: KPopTree, n_snps: int, seed: Optional[int] = None) -> np.ndarray:
    """Repeat `one_mutation_from_tree` n_snps times to form an (n_total × n_snps) matrix."""
    rng = np.random.default_rng(seed)
    G = np.zeros((tree.n_total, n_snps), dtype=int)
    for j in range(n_snps):
        G[:, j] = one_mutation_from_tree(tree, rng)
    return G


# ------------------------------
# End-to-end convenience wrapper matching your CSV shape
# ------------------------------

def simulate_kpop_binary_genotypes(
    k: int,
    c: int,
    L: int,
    M: float,
    seed: Optional[int] = None,
    return_pop: bool = True,
):
    """Simulate a k×k grid with c samples per population, generate L 0/1 SNPs.

    Returns a pandas.DataFrame whose columns are V1..VL and, if `return_pop`, a
    last column 'populations' (1..k*k repeated c times) — i.e., what your
    notebook expects before converting to inbred genotypes with F.
    """
    import pandas as pd  # delayed import

    tree = simulate_tree_kpop_lattice(k=k, n_per_pop=c, M=M, seed=seed)
    G = simulate_genotype_matrix_from_tree(tree, n_snps=L, seed=seed)

    cols = {f"V{j+1}": G[:, j] for j in range(L)}
    df = pd.DataFrame(cols)
    if return_pop:
        df["populations"] = [p + 1 for p in range(k * k) for _ in range(c)]
    return df


# ------------------------------
# Optional: vanilla (unstructured) coalescent without recombination
# ------------------------------
# This mirrors the no-recombination branch of `simulate.coalescent` in coalescent.r
# and returns a (sample × S) 0/1 matrix with S ≈ Poisson(theta * total tree length / 2)
# together with the per-mutation positions in (0,1].

@dataclass
class SimpleCoalescent:
    tree_time: np.ndarray        # (2n-1,)
    D1: np.ndarray               # (2n-1,)
    D2: np.ndarray               # (2n-1,)
    branch_length: np.ndarray    # (2n-1,)
    parent: np.ndarray           # (2n-1,)
    genotypes: np.ndarray        # (n × S) 0/1
    positions: np.ndarray        # (S,) in (0,1]


def simulate_coalescent_no_recomb(sample: int, theta: float, seed: Optional[int] = None) -> SimpleCoalescent:
    """Kingman coalescent for `sample` haplotypes, no recombination, infinite-sites mutations.

    Closely follows the no-recombination branch in `simulate.coalescent` (R).
    """
    rng = np.random.default_rng(seed)

    n = sample
    n_nodes = 2 * n - 1
    time = np.zeros(n_nodes)
    D1 = np.full(n_nodes, -1, dtype=int)
    D2 = np.full(n_nodes, -1, dtype=int)
    parent = np.full(n_nodes, -1, dtype=int)
    branch_length = np.zeros(n_nodes)

    # Build the ranked tree (times)
    k = n
    klist = list(range(n))
    t = 0.0
    current = n

    while k > 1:
        rate = k * (k - 1) / 2.0
        dt = -math.log(rng.random()) / rate
        t += dt
        # pick two lineages uniformly
        i = int(rng.integers(0, k))
        # swap to end
        klist[i], klist[k - 1] = klist[k - 1], klist[i]
        j = int(rng.integers(0, k - 1))
        # add new internal node
        time[current] = t
        D1[current] = klist[k - 1]
        D2[current] = klist[j]
        parent[klist[k - 1]] = current
        parent[klist[j]] = current
        branch_length[klist[k - 1]] = t - time[klist[k - 1]]
        branch_length[klist[j]] = t - time[klist[j]]
        # replace j with current
        klist[j] = current
        current += 1
        k -= 1
        klist = klist[:k]

    # Total number of mutations across the tree ~ Poisson(theta * total_length / 2)
    total_length = float(branch_length[: current - 1].sum())
    n_mut = rng.poisson(theta * total_length / 2.0)

    # Drop those mutations uniformly along branches (record positions just as uniforms)
    positions = np.sort(rng.random(n_mut))
    G = np.zeros((n, n_mut), dtype=int)
    for _ in range(n_mut):
        node = _sample_weighted(branch_length, rng)
        _fill_in_genotype_recursive(n, G[:, _], D1, D2, node)

    return SimpleCoalescent(
        tree_time=time,
        D1=D1,
        D2=D2,
        branch_length=branch_length,
        parent=parent,
        genotypes=G,
        positions=positions,
    )
