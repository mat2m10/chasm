
"""
kpop.py — Lattice-structured coalescent + SMC utilities (Python port of kpop.r)

This module provides functions to simulate genealogy trees for samples drawn
from a k x k grid of demes (populations) with nearest-neighbour migration,
and to perform Sequentially Markov Coalescent (SMC) prune–regraft updates.

Design notes
------------
- Python (0-based) indexing is used throughout.
  * Leaves are nodes 0..(n-1)
  * Internal nodes are n..(2n-2)
  * MRCA index is (2n-2)
  * Node IDs equal their row index.
- The tree is represented as a NumPy array with named columns via indices:
    COL_NODE_ID = 0
    COL_TIME    = 1
    COL_D1      = 2
    COL_D2      = 3
    COL_MUT     = 4
  After `annotate_tree`, two columns are appended:
    COL_BLEN    = 5  (branch length to parent)
    COL_PARENT  = 6  (parent node index, or -1 for root/MRCA)
- `annot` is a list (length = number of nodes) of arrays with shape (2, m),
  where each column is [pop_id, time_of_entry]. For leaves we start with
  [[initial_pop],[0.0]]. Internal nodes can record their creation time/pop.

This port is faithful to the original R code's behaviour while fixing a few
minor bugs (e.g., missing local nseq in lineages_time; missing find.descendants).

Dependencies: numpy, matplotlib (for plotting)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

# Column indices for the tree array
COL_NODE_ID = 0
COL_TIME    = 1
COL_D1      = 2
COL_D2      = 3
COL_MUT     = 4
# Appended by annotate_tree
COL_BLEN    = 5
COL_PARENT  = 6


# -------------------------------
# Utility / helper functionality
# -------------------------------

def decode_m(m: int, k: int = 2) -> Tuple[int, int]:
    """
    Decode a 1-based lattice index m -> (i, j) using k x k grid (both 1-based).
    The Python port uses 0-based pop indices elsewhere, but this helper
    mirrors the R function's contract.
    """
    if m < 1 or m > k * k:
        raise ValueError("m must satisfy 1 <= m <= k^2")
    i = (m - 1) // k + 1
    j = m - (i - 1) * k
    return i, j


def decode_cell_counts(k: int, n_sim: int) -> np.ndarray:
    """
    For a list of k*k cell counts return positions (x,y) for each cell,
    repeating each position n_sim times (matching the R behaviour).
    """
    coords = np.array([decode_m(m + 1, k) for m in range(k * k)], dtype=int)  # 1-based decode
    x = np.repeat(coords[:, 0], n_sim)  # still 1-based
    y = np.repeat(coords[:, 1], n_sim)
    return np.column_stack((x, y))


def sample_individuals(k: int, n_tot: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample n_tot individuals over k^2 demes (with replacement), returning
    counts per deme as length-k^2 vector.
    """
    rng = np.random.default_rng() if rng is None else rng
    positions = rng.integers(0, k * k, size=n_tot)  # 0..k^2-1
    counts = np.bincount(positions, minlength=k * k)
    return counts


def _inorder_leaf_order(tree: np.ndarray) -> List[int]:
    """
    Compute an in-order left-to-right leaf order for plotting.
    Returns list of leaf indices (0..n-1).
    """
    n = (len(tree) + 1) // 2
    root = len(tree) - 1

    order: List[int] = []

    def dfs(u: int):
        d1 = int(tree[u, COL_D1])
        d2 = int(tree[u, COL_D2])
        # leaf if d1 == -1 (by construction leaves have children -1)
        if d1 < 0 and d2 < 0:
            if u < n:
                order.append(u)
            return
        if d1 >= 0:
            dfs(d1)
        if d2 >= 0:
            dfs(d2)

    dfs(root)
    # Ensure only leaves
    return [i for i in order if i < n]


def annotate_tree(tree: np.ndarray) -> np.ndarray:
    """
    Add Branch.length and Parent columns to the tree array.
    Assumes internal nodes are n..2n-2 and leaves have COL_D1==COL_D2==-1.
    """
    n_nodes = len(tree)
    n = (n_nodes + 1) // 2

    # Extend the array by two columns
    out = np.concatenate([tree, np.zeros((n_nodes, 2), dtype=float)], axis=1)
    out[:, COL_PARENT] = -1  # initialize as -1 (root and any unassigned)

    for j in range(n, n_nodes):
        d1 = int(out[j, COL_D1])
        d2 = int(out[j, COL_D2])
        if d1 >= 0:
            out[d1, COL_PARENT] = j
            out[d1, COL_BLEN] = out[j, COL_TIME] - out[d1, COL_TIME]
        if d2 >= 0:
            out[d2, COL_PARENT] = j
            out[d2, COL_BLEN] = out[j, COL_TIME] - out[d2, COL_TIME]

    # Root/MRCA has parent -1 and branch length 0
    out[-1, COL_BLEN] = 0.0
    out[-1, COL_PARENT] = -1
    return out


def tree_length_below_node(tree: np.ndarray, node: int, t: float = 0.0) -> float:
    """
    Total branch length below a node (recursive), mirroring Tree.length in R.
    Requires annotate_tree to have been run (for times and structure).
    """
    d1 = int(tree[node, COL_D1])
    d2 = int(tree[node, COL_D2])
    if d1 < 0 and d2 < 0:
        return t
    if d1 >= 0:
        t += tree[node, COL_TIME] - tree[d1, COL_TIME]
        t = tree_length_below_node(tree, d1, t)
    if d2 >= 0:
        t += tree[node, COL_TIME] - tree[d2, COL_TIME]
        t = tree_length_below_node(tree, d2, t)
    return t


def lineages_time(tree: np.ndarray, time: float) -> List[int]:
    """
    List lineages (node indices) present at a given time point (just below any
    coalescent event). Mirrors the logic in R but fixed to compute n locally.
    """
    n = (len(tree) + 1) // 2
    mrca = len(tree) - 1
    present: List[int] = []
    # Iterate internal nodes (newest upwards), include children alive at 'time'
    for i in range(mrca, n - 1, -1):
        if tree[i, COL_TIME] > time:
            d1 = int(tree[i, COL_D1])
            d2 = int(tree[i, COL_D2])
            if tree[d1, COL_TIME] < time:
                present.append(d1)
            if tree[d2, COL_TIME] < time:
                present.append(d2)
    return present


def find_descendants_leaves(tree: np.ndarray, node: int) -> List[int]:
    """
    Return all descendant leaves (indices < n) of 'node' (inclusive of leaves only).
    """
    n = (len(tree) + 1) // 2
    res: List[int] = []
    d1 = int(tree[node, COL_D1])
    d2 = int(tree[node, COL_D2])
    if d1 < 0 and d2 < 0:
        if node < n:
            return [node]
        return []
    if d1 >= 0:
        res.extend(find_descendants_leaves(tree, d1))
    if d2 >= 0:
        res.extend(find_descendants_leaves(tree, d2))
    return res


# ----------------------------------
# Lattice coalescent with migration
# ----------------------------------

@dataclass
class SimResult:
    tree: np.ndarray          # shape (2n-1, 5) or (2n-1, 7) after annotate_tree
    annot: List[np.ndarray]   # per-node 2 x m array: [[pop_ids...],[times...]]


def simulate_tree_kpop_lattice(
    k: int = 2,
    n_per_pop: Sequence[int] | None = None,
    M: float = 1.0,
    rng: np.random.Generator | None = None,
) -> SimResult:
    """
    Simulate a time-ordered coalescent tree for samples on a k x k lattice
    with nearest-neighbour migration at rate M (same conventions as R code).

    Parameters
    ----------
    k : grid dimension (k x k demes)
    n_per_pop : iterable of length k^2 giving sample counts per deme
                (default: 10 per deme).
    M : migration rate scalar (M == 0 not allowed; M > 10 is clipped to 10)
    rng : optional NumPy Generator (for reproducibility)

    Returns
    -------
    SimResult(tree, annot)
      tree: array columns [Node.ID, Time, D1, D2, Mutations]
      annot: list of 2 x m arrays per node, recording (deme, time)
    """
    rng = np.random.default_rng() if rng is None else rng
    if n_per_pop is None:
        n_per_pop = [10] * (k * k)
    n_per_pop = np.array(n_per_pop, dtype=int)
    if n_per_pop.size != k * k:
        raise ValueError("n_per_pop must have length k^2")
    if M == 0:
        raise ValueError("M == 0 is not allowed")
    if M > 10:
        M = 10.0

    n_pop = k * k
    n_tot = int(n_per_pop.sum())
    n_nodes = 2 * n_tot - 1

    # Tree base matrix: Node.ID, Time, D1, D2, Mutations
    tree = np.zeros((n_nodes, 5), dtype=float)
    tree[:, COL_NODE_ID] = np.arange(n_nodes)  # IDs equal indices
    tree[:n_tot, COL_D1:COL_D2+1] = -1  # leaves have no children

    current_node = n_tot  # next internal node index to fill
    # active lineages (node indices) and their deme memberships (0..n_pop-1)
    k_list = list(range(n_tot))
    pop_list = np.repeat(np.arange(n_pop, dtype=int), n_per_pop).tolist()

    # Migration adjacency (4-neighbourhood; -1 means no neighbour)
    mig_mat = -np.ones((n_pop, 4), dtype=int)
    for i in range(k):
        for j in range(k):
            m = i * k + j  # 0-based deme id
            # up, down, left, right (matching R order)
            if i > 0:     mig_mat[m, 0] = (i - 1) * k + j
            if i < k - 1: mig_mat[m, 1] = (i + 1) * k + j
            if j > 0:     mig_mat[m, 2] = i * k + (j - 1)
            if j < k - 1: mig_mat[m, 3] = i * k + (j + 1)

    mig_mask = mig_mat >= 0
    mig_rates = mig_mask.sum(axis=1).astype(float)  # number of neighbours per deme

    # Row-normalized migration choice probabilities among the available neighbours
    mig_prob = np.where(mig_mask, 1.0, 0.0)
    mig_prob = (mig_prob.T / mig_prob.sum(axis=1)).T

    # per-deme active counts (mutable)
    n_active = n_per_pop.astype(float).copy()

    # Annotations: per-node 2 x m array
    annot: List[np.ndarray] = []
    for i in range(n_nodes):
        if i < n_tot:
            pop = pop_list[i]
            annot.append(np.array([[pop], [0.0]], dtype=float))
        else:
            annot.append(np.zeros((2, 1), dtype=float))

    t = 0.0
    while n_active.sum() > 1.0:
        # per-deme coalescent rates
        rate_co = n_active * (n_active - 1.0) / 2.0
        rate_co_tot = rate_co.sum()

        # per-deme migration rates (M/2 * n_i * (#neighbours))
        rate_mig = (M / 2.0) * n_active * mig_rates
        rate_mig_tot = rate_mig.sum()

        # next event time
        t_event = -math.log(rng.random()) / (rate_co_tot + rate_mig_tot)
        t += t_event

        # migration or coalescent?
        if rng.random() <= rate_mig_tot / (rate_mig_tot + rate_co_tot):
            # Migration event: pick a deme proportional to its migration rate
            deme = rng.choice(n_pop, p=rate_mig / rate_mig_tot)
            # choose a lineage currently in that deme
            indices = [idx for idx, d in enumerate(pop_list) if d == deme]
            if not indices:
                # numeric safeguard; fall back to any lineage
                i_idx = rng.integers(0, len(pop_list))
            else:
                i_idx = rng.choice(indices)
            # choose neighbour
            neighs = mig_mat[deme]
            probs = mig_prob[deme]
            # restrict to valid neighbours
            valid = neighs >= 0
            neigh = rng.choice(neighs[valid], p=(probs[valid] / probs[valid].sum()))
            # update counts and assignment
            n_active[deme] -= 1.0
            n_active[neigh] += 1.0
            pop_list[i_idx] = int(neigh)
            # append annotation to that lineage's node record
            node_id = k_list[i_idx]
            annot[node_id] = np.column_stack([annot[node_id], np.array([neigh, t], dtype=float)])
        else:
            # Coalescent within a deme
            deme = rng.choice(n_pop, p=rate_co / rate_co_tot)
            # pick two lineages from that deme
            candidates = [idx for idx, d in enumerate(pop_list) if d == deme]
            if len(candidates) < 2:
                # numeric safeguard; skip (should be rare)
                continue
            pair = rng.choice(candidates, size=2, replace=False)
            i1, i2 = int(pair[0]), int(pair[1])

            # create internal node
            tree[current_node, COL_TIME] = t
            tree[current_node, COL_D1] = k_list[i1]
            tree[current_node, COL_D2] = k_list[i2]

            # replace min(index) by new node, drop the other by swapping with last
            a, b = sorted([i1, i2])
            k_list[a] = current_node
            # move last into position b
            k_list[b] = k_list[-1]
            pop_list[b] = pop_list[-1]

            # shrink active lists by 1
            k_list.pop()
            pop_list.pop()
            n_active[deme] -= 1.0

            # annotate the new internal node (its deme and time of creation)
            annot[current_node] = np.array([[deme], [t]], dtype=float)

            current_node += 1

    # Set any unset leaves' times to 0 (already zero), mutations to 0
    # Return the raw (un-annotated) tree and annotations
    return SimResult(tree=tree, annot=annot)


# -----------------------
# SMC / recombination ops
# -----------------------

def sample_rec_event(tree: np.ndarray, branch_length_col: int = COL_BLEN) -> int:
    """
    Choose a branch for a recombination event with probability proportional
    to its branch length.
    Requires annotate_tree to have been run.
    """
    bl = tree[:, branch_length_col].copy()
    bl[bl < 0] = 0
    p = bl / bl.sum()
    return int(np.random.default_rng().choice(len(tree), p=p))


def choose_rec_time(tree: np.ndarray, which_branch: int, parent_col: int = COL_PARENT) -> float:
    """
    Sample a time uniformly along the chosen branch (between node time and parent time).
    """
    parent = int(tree[which_branch, parent_col])
    if parent < 0:
        # root: choose a small epsilon above node time
        return float(tree[which_branch, COL_TIME])
    t0 = float(tree[which_branch, COL_TIME])
    t1 = float(tree[parent, COL_TIME])
    u = np.random.default_rng().random()
    return t0 + u * (t1 - t0)


@dataclass
class CoalescentEvent:
    who_co: int   # node index to coalesce with
    t_co: float   # time of coalescence


def choose_co_event(tree: np.ndarray, which_branch: int, t_rec: float, debug: bool = False) -> CoalescentEvent:
    """
    Sample a coalescent event for a floating lineage (basic SMC approximation).
    Mirrors the R logic with epochs defined at internal node times.
    """
    n = (len(tree) + 1) // 2
    mrca = len(tree) - 1

    # lineages present at t_rec excluding the recombined lineage
    lins = [x for x in lineages_time(tree, t_rec) if x != which_branch]
    n_lin = len(lins)
    epoch = 2 * n - len(lins)

    is_free = True
    t_curr = t_rec
    rng = np.random.default_rng()

    if debug:
        print(f"Sampling coalescent for branch {which_branch} at t_rec={t_rec:.6f}")

    while is_free:
        # exponential waiting time with rate = n_lin (coalescence with any of the lins)
        t_co = -math.log(rng.random()) / max(n_lin, 1)
        t_curr += t_co

        if epoch < len(tree) and t_curr < tree[epoch, COL_TIME]:
            # coalescence before next tree event
            is_free = False
            if n_lin > 1:
                who_co = int(rng.choice(lins))
            else:
                who_co = int(lins[0])
            if debug:
                print(f"Coalescence with {who_co} at t={t_curr:.6f}")
        else:
            # advance to next coalescent in the fixed tree
            if epoch >= len(tree):
                # past MRCA: coalesce above the root
                t_curr += -math.log(rng.random())
                who_co = mrca
                is_free = False
                if debug:
                    print("Reached MRCA; sampling above root")
            else:
                t_curr = float(tree[epoch, COL_TIME])
                # update lineage set: remove two daughters, add their parent (epoch)
                d1 = int(tree[epoch, COL_D1]); d2 = int(tree[epoch, COL_D2])
                lins = [x for x in lins if x not in (d1, d2)]
                lins.append(epoch)
                n_lin = len(lins)
                if debug:
                    print(f"Next fixed-tree coalescent {d1}^{d2} at t={t_curr:.6f}")
                epoch += 1

    return CoalescentEvent(who_co=who_co, t_co=t_curr)


def _remap_after_sort(tree2: np.ndarray) -> np.ndarray:
    """
    After reordering rows by time, restore Node.IDs to 0..m-1 and
    remap D1, D2, Parent to new indices.
    """
    m = len(tree2)
    # argsort by time, stable to preserve leaves (time 0) ordering somewhat
    order = np.argsort(tree2[:, COL_TIME], kind='mergesort')
    tree_sorted = tree2[order].copy()
    # OldID -> NewID map
    new_id = np.empty(m, dtype=int)
    new_id[order] = np.arange(m)

    # Remap columns
    for col in (COL_D1, COL_D2, COL_PARENT):
        vals = tree_sorted[:, col].astype(int)
        mask = vals >= 0
        vals[mask] = new_id[vals[mask]]
        tree_sorted[:, col] = vals

    # Reset Node.IDs to match row index
    tree_sorted[:, COL_NODE_ID] = np.arange(m)

    # Fix root annotations
    tree_sorted[-1, COL_BLEN] = 0.0
    tree_sorted[-1, COL_PARENT] = -1

    return tree_sorted


def prune_regraft(tree: np.ndarray, which_branch: int, t_rec: float, who_co: int, t_co: float, debug: bool = False) -> np.ndarray:
    """
    Prune the branch 'which_branch' at time t_rec and regraft to coalesce with 'who_co'
    at time t_co, producing a new time-ordered tree. Port of prune.regraft in R.
    Requires an annotated tree (Branch.length, Parent).
    """
    n = (len(tree) + 1) // 2
    mrca = len(tree) - 1

    if debug:
        print(f"Prune-regraft: move node {which_branch} @ {t_rec:.6f} to coalesce with {who_co} @ {t_co:.6f}")

    # Identify parent (pn), grandparent (gpn), and sister of which_branch
    pn = int(tree[which_branch, COL_PARENT])
    if pn < 0:
        # recombination on root branch; nothing to do
        return tree.copy()
    d1 = int(tree[pn, COL_D1]); d2 = int(tree[pn, COL_D2])
    sister = d2 if d1 == which_branch else d1

    gpn = int(tree[pn, COL_PARENT]) if pn != mrca else pn
    dd2_right = 1 if int(tree[gpn, COL_D2]) == pn else 0  # which daughter of gpn is pn

    # Parent of 'who_co' (special case if coalescing with sister -> parent is gpn)
    pn_co = int(tree[who_co, COL_PARENT])
    if who_co == sister:
        pn_co = gpn
    if who_co != mrca:
        dd3_right = 1 if int(tree[pn_co, COL_D2]) == (pn if who_co == sister else who_co) else 0

    tree2 = tree.copy()

    # Remove parental node: connect gpn directly to sister
    if gpn >= 0:
        if dd2_right == 1:
            tree2[gpn, COL_D2] = sister
        else:
            tree2[gpn, COL_D1] = sister
        tree2[sister, COL_PARENT] = gpn
        tree2[sister, COL_BLEN] = tree2[gpn, COL_TIME] - tree2[sister, COL_TIME]

    # If coalescing with your own parent, treat as coalescing with your sister
    if pn == who_co:
        who_co = sister
        pn_co = gpn

    # Insert new node (reuse pn) below parent of 'who_co'
    tree2[who_co, COL_PARENT] = pn
    if pn_co >= 0:
        if dd3_right == 1:
            tree2[pn_co, COL_D2] = pn
        else:
            tree2[pn_co, COL_D1] = pn

    # Now define the new node 'pn'
    tree2[pn, COL_TIME] = t_co
    tree2[pn, COL_D1] = which_branch
    tree2[pn, COL_D2] = who_co
    tree2[pn, COL_PARENT] = pn_co
    tree2[pn, COL_BLEN] = (tree2[pn_co, COL_TIME] - tree2[pn, COL_TIME]) if pn_co >= 0 else 0.0

    # Update children parents and branch lengths
    tree2[which_branch, COL_PARENT] = pn
    tree2[who_co, COL_PARENT] = pn
    tree2[which_branch, COL_BLEN] = tree2[pn, COL_TIME] - tree2[which_branch, COL_TIME]
    tree2[who_co, COL_BLEN] = tree2[pn, COL_TIME] - tree2[who_co, COL_TIME]

    # Reorder by time and remap indices
    tree2 = _remap_after_sort(tree2)

    return tree2


def simulate_next_tree_smc(tree: np.ndarray, show_tree: bool = False, debug: bool = False) -> np.ndarray:
    """
    Take an annotated tree and simulate a single SMC step:
      1) sample recombination branch (∝ branch length),
      2) sample recombination time on that branch,
      3) sample coalescent partner/time for floating lineage,
      4) prune & regraft to obtain new tree.
    """
    # choose branch
    b = sample_rec_event(tree, branch_length_col=COL_BLEN)
    # time on that branch
    t_rec = choose_rec_time(tree, which_branch=b, parent_col=COL_PARENT)
    # choose coalescent event
    evt = choose_co_event(tree, which_branch=b, t_rec=t_rec, debug=debug)
    # prune & regraft
    tree2 = prune_regraft(tree, which_branch=b, t_rec=t_rec, who_co=evt.who_co, t_co=evt.t_co, debug=debug)
    if show_tree:
        plot_tree(tree2)
    return tree2


def check_tree(tree: np.ndarray, verbose: bool = False) -> bool:
    """
    Basic structural checks: monotonic times, child-parent consistency,
    branch length consistency. Returns True if OK, False otherwise.
    """
    ok = True
    # times monotonic
    if np.any(np.diff(tree[:, COL_TIME]) < -1e-12):
        if verbose: print("Error: times are not monotonic")
        ok = False
    n = (len(tree) + 1) // 2
    m = len(tree) - 1
    # descendants have correct parent
    for i in range(n, m + 1):
        d1 = int(tree[i, COL_D1]); d2 = int(tree[i, COL_D2])
        if d1 >= 0 and int(tree[d1, COL_PARENT]) != i: 
            if verbose: print("Error: descendant-parent mismatch")
            ok = False
        if d2 >= 0 and int(tree[d2, COL_PARENT]) != i: 
            if verbose: print("Error: descendant-parent mismatch")
            ok = False
    # branch lengths consistent
    for i in range(0, m):
        p = int(tree[i, COL_PARENT])
        if p >= 0:
            tl = tree[p, COL_TIME] - tree[i, COL_TIME]
            if abs(tl - tree[i, COL_BLEN]) > 1e-6:
                if verbose: print(f"Error: branch length mismatch at node {i}")
                ok = False
    return ok


# -----------------
# Mutation routines
# -----------------

def choose_mutation(tree: np.ndarray, uniform: bool = True, n_mutations: int = 1, annot: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """
    Choose node indices for mutations.
    If uniform=True, sample proportional to branch lengths (1/f SFS).
    Else, mutations accrue at nodes proportional to 2^{#annot columns} (R quirk).
    """
    rng = np.random.default_rng()
    nr = len(tree)
    if uniform or annot is None:
        bl = np.maximum(tree[:, COL_BLEN], 0.0)
        p = bl / bl.sum()
    else:
        pr = np.array([a.shape[1] for a in annot], dtype=float)
        pr = np.power(2.0, pr)
        pr[-1] = 0.0  # root gets weight 0
        p = pr / pr.sum()
    return rng.choice(np.arange(nr), size=n_mutations, replace=True, p=p)


def add_mutation(tree: np.ndarray, node: int) -> np.ndarray:
    """
    Return a 0/1 vector of derived allele presence across leaves (length n),
    for a mutation placed at 'node' on the tree.
    """
    n = (len(tree) + 1) // 2
    seq = np.zeros(n, dtype=int)
    leaves = find_descendants_leaves(tree, node)
    seq[leaves] = 1
    return seq


def summarise_lattice(seq: np.ndarray, k: int = 1, n_per_pop: Optional[Sequence[int]] = None) -> dict:
    """
    Summarise a mutation's spatial distribution over the lattice.
    Returns { 'x_bar': total derived count, 'fst': Fst-like summary }.
    """
    if n_per_pop is None:
        n_per_pop = [len(seq)]
    n_per_pop = np.array(n_per_pop, dtype=int)
    n_tot = int(n_per_pop.sum())
    if len(seq) != n_tot:
        raise ValueError("Length of seq must equal total sample size")

    # per-deme means
    deme_ids = np.repeat(np.arange(k * k), n_per_pop)
    lfreq = np.array([seq[deme_ids == d].mean() for d in range(k * k)])
    mn = seq.mean()

    # within-deme and total pairwise diversities
    withins = 2 * lfreq * (1 - lfreq) * (n_per_pop / (n_per_pop - 1).clip(min=1))  # handle n=1 gracefully
    pwd_w = np.nanmean(np.where(n_per_pop > 1, withins, np.nan))
    pwd_t = 2 * mn * (1 - mn) * (n_tot / (max(n_tot - 1, 1)))
    fst = 1.0 - (pwd_w / pwd_t) if pwd_t > 0 else 0.0

    return {'x_bar': int(seq.sum()), 'fst': float(fst)}


# --------------
# Plotting utils
# --------------

def color_map_2d(n: int) -> List[str]:
    """
    Build an n-by-n 2D color map (flattened length n*n) roughly analogous to the R function.
    Returns list of hex strings usable in matplotlib.
    """
    # Base gradients (red->blue and black->green) in 0..1 RGB
    x = np.linspace(0, 1, n)
    red_blue = np.column_stack([1 - x, 0 * x + 0.1, x])  # tweak for contrast
    black_green = np.column_stack([0 * x, 0.3 * x, 0.2 + 0.8 * x])

    fader = np.abs(np.linspace(-1, 1, n))
    fader2 = np.linspace(1, 0, n) ** 2
    fader3 = np.sqrt(np.linspace(0, 1, n))

    cols = np.zeros((n * n, 3))
    idx = 0
    for i in range(n):
        for j in range(n):
            v = np.maximum((1 - fader2[j]) * fader[i] * red_blue[i] + (fader2[j]) * red_blue[i],
                           fader3[j] * black_green[j])
            cols[idx] = np.clip(v, 0, 1)
            idx += 1

    def to_hex(rgb):
        return '#%02x%02x%02x' % tuple((rgb * 255).astype(int))

    return [to_hex(c) for c in cols]


def plot_tree(tree: np.ndarray, annot: Optional[List[np.ndarray]] = None, k: int = 1,
              names_plot: bool = True, linewidth: float = 1.0, add_internal: bool = True,
              add_mutations: bool = False, cex_mut: float = 20.0, ax: Optional[plt.Axes] = None):
    """
    Plot a time-ordered tree; if annot and k>1 are provided, colour branches by deme.
    """
    if ax is None:
        fig, ax = plt.subplots()

    n = (len(tree) + 1) // 2
    mrca = len(tree) - 1
    order = _inorder_leaf_order(tree)
    xpos = np.empty(len(tree), dtype=float)
    for i in range(n):
        xpos[i] = order.index(i) + 1
    # iteratively set internal node x positions as midpoints
    for node in range(n, mrca + 1):
        d1 = int(tree[node, COL_D1]); d2 = int(tree[node, COL_D2])
        xpos[node] = 0.5 * (xpos[d1] + xpos[d2])

    # choose colour palette per deme if given
    branch_color = 'black'
    pop_pals = None
    if annot is not None and k > 1:
        pop_pals = color_map_2d(int(math.sqrt(k * k)))

    # draw branches
    ax.set_ylim(0, tree[mrca, COL_TIME] * 1.05 if tree[mrca, COL_TIME] > 0 else 1.0)
    ax.set_xlim(0, n + 1)
    ax.set_ylabel("Time")
    ax.set_xticks([])

    for node in range(n, mrca + 1):
        d1 = int(tree[node, COL_D1]); d2 = int(tree[node, COL_D2])
        # horizontal
        col = branch_color
        if pop_pals is not None and annot is not None:
            deme = int(annot[node][0, 0])
            col = pop_pals[deme]
        ax.hlines(tree[node, COL_TIME], xpos[d1], xpos[d2], colors=col, linewidth=linewidth)
        # verticals
        ax.vlines(xpos[d1], tree[d1, COL_TIME], tree[node, COL_TIME], colors=col, linewidth=linewidth)
        ax.vlines(xpos[d2], tree[d2, COL_TIME], tree[node, COL_TIME], colors=col, linewidth=linewidth)
        if add_internal:
            ax.text(xpos[node], tree[node, COL_TIME], str(node), ha='center', va='bottom', fontsize=8)

    if names_plot:
        for i in range(n):
            ax.text(xpos[i], 0, f"{i}", rotation=90, va='top', ha='center', fontsize=8)

    if add_mutations:
        # draw mutation dots proportional along branches with COL_MUT counts (optional feature)
        for node in range(n, mrca + 1):
            for d in (int(tree[node, COL_D1]), int(tree[node, COL_D2])):
                if tree[d, COL_MUT] > 0:
                    y0, y1 = tree[d, COL_TIME], tree[node, COL_TIME]
                    ys = y0 + np.random.random(int(tree[d, COL_MUT])) * (y1 - y0)
                    ax.scatter([xpos[d]] * len(ys), ys, s=cex_mut)

    ax.set_title("Coalescent tree (k-pop lattice)")
    if ax is None:
        plt.show()


def plot_mutation(seq: np.ndarray, k: int = 1, n_per_pop: Optional[Sequence[int]] = None,
                  n_bins: int = 10, edge_color: str = "black", ax: Optional[plt.Axes] = None,
                  override_palette: Optional[Sequence[str]] = None):
    """
    Heatmap-like plot of a mutation's frequency across the lattice.
    """
    if n_per_pop is None:
        n_per_pop = [len(seq)]
    n_per_pop = np.array(n_per_pop, dtype=int)
    n_tot = int(n_per_pop.sum())
    if len(seq) != n_tot:
        raise ValueError("Length of seq must equal total sample size")

    if ax is None:
        fig, ax = plt.subplots()

    # per-deme means
    deme_ids = np.repeat(np.arange(k * k), n_per_pop)
    lfreq = np.array([seq[deme_ids == d].mean() for d in range(k * k)])

    # color scale
    if override_palette is None:
        pal = plt.get_cmap("YlOrRd")
        # normalize to [0,1]
        colors = [pal(v) for v in lfreq.clip(0, 1)]
    else:
        colors = override_palette

    # draw grid
    for i in range(k):
        for j in range(k):
            idx = i * k + j
            rect = plt.Rectangle((i, j), 1, 1, facecolor=colors[idx], edgecolor=edge_color)
            ax.add_patch(rect)

    ax.set_xlim(0, k)
    ax.set_ylim(0, k)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Spatial mutation frequency")

    if ax is None:
        plt.show()


# ----------------------
# Minimal usage example
# ----------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Example: k=3, 3x3 grid, 4 samples per deme
    k = 3
    n_per_pop = [4] * (k * k)
    sim = simulate_tree_kpop_lattice(k=k, n_per_pop=n_per_pop, M=0.5, rng=rng)

    # Annotate tree for SMC operations
    tree_a = annotate_tree(sim.tree)

    # Plot the initial tree (optional)
    # plot_tree(tree_a, annot=sim.annot, k=k)

    # Place a mutation and visualise
    mut_node = int(choose_mutation(tree_a, uniform=True, n_mutations=1, annot=sim.annot)[0])
    seq = add_mutation(tree_a, mut_node)
    summary = summarise_lattice(seq, k=k, n_per_pop=n_per_pop)
    print("Mutation summary:", summary)

    # One SMC step
    tree_next = simulate_next_tree_smc(tree_a, show_tree=False, debug=False)
    print("Tree valid after SMC:", check_tree(tree_next, verbose=True))
