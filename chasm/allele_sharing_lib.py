"""
Functions
---------
- sample_distance(i, n, k)
- sample_distance_matrix(n, k)
- allele_sharing_by_distance(gt, n, k)
- plot_excess_allele_sharing(sr, labels=("<0.04","0.04-0.1",">0.1"), log_scale=False, **kwargs)

Notes
-----
Assumes individuals are ordered as k*k*n, where for each lattice position (k x k grid)
there are n individuals. Indexing follows the original R code logic using 1-based
indices internally when mapping to positions, but the Python interface uses 0-based
arrays as usual.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "sample_distance",
    "sample_distance_matrix",
    "allele_sharing_by_distance",
    "plot_excess_allele_sharing",
]


def _decode_m(idx: int, k: int) -> np.ndarray:
    """Map a 1-based lattice index (1..k*k) to 2D coordinates on a k x k grid.

    Returns zero-based coordinates [row, col]. The exact row/col orientation is
    arbitrary as long as it's consistent across calls; the Manhattan distances are
    invariant to swapping axes.
    """
    if not (1 <= idx <= k * k):
        raise ValueError("idx must be within 1..k*k")
    idx0 = idx - 1
    row = idx0 // k
    col = idx0 % k
    return np.array([row, col], dtype=int)


def sample_distance(i: int, n: int, k: int) -> np.ndarray:
    """Compute Manhattan distance from individual *i* to all others.

    Parameters
    ----------
    i : int
        1-based index of the focal individual (1..n*k*k) to match the R version.
    n : int
        Number of individuals per lattice position.
    k : int
        Lattice side length (k x k positions).

    Returns
    -------
    np.ndarray
        Vector of length n*k*k of Manhattan distances from individual i to each individual.
    """
    nkk = n * k * k
    if not (1 <= i <= nkk):
        raise ValueError("i must be within 1..n*k*k (1-based index)")

    # Map individuals to lattice positions via ceiling(index / n)
    i_pos = _decode_m(math.ceil(i / n), k)
    # Precompute positions for all indices 1..nkk
    pos_indices = [math.ceil(q / n) for q in range(1, nkk + 1)]
    all_pos = np.vstack([_decode_m(pi, k) for pi in pos_indices])  # shape (nkk, 2)

    d = np.abs(all_pos - i_pos).sum(axis=1)  # Manhattan distance
    return d.astype(int)


def sample_distance_matrix(n: int, k: int) -> np.ndarray:
    """Compute the full pairwise Manhattan distance matrix among n*k*k individuals.

    Returns an (nkk x nkk) integer matrix.
    """
    nkk = n * k * k
    # Positions for each individual (1..nkk)
    pos_indices = np.array([math.ceil(q / n) for q in range(1, nkk + 1)], dtype=int)
    coords = np.vstack([_decode_m(int(pi), k) for pi in pos_indices])  # (nkk, 2)

    # Efficient pairwise Manhattan distance using broadcasting
    # |x1-x2| + |y1-y2|
    diff = np.abs(coords[:, None, :] - coords[None, :, :])  # (nkk, nkk, 2)
    dm = diff.sum(axis=2).astype(int)
    return dm


def allele_sharing_by_distance(gt: np.ndarray, n: int, k: int) -> np.ndarray:
    """For rare, low-frequency, and common alleles, compute excess sharing by distance.

    This mirrors the R function `allele.sharing.by.distance`. Given a genotype matrix
    `gt` of shape (n*k*k, L), with entries 0/1 indicating presence of the allele in each
    individual at locus L, it computes, for each locus, the fraction of *ordered* pairs
    (i<j with gt[i,m]==1) where the partner j also carries the allele, stratified by
    lattice distance. The locus-specific curves are then divided by the derived allele
    frequency (column mean), and finally averaged within frequency bins (<0.04, 0.04-0.1,
    >0.1) to produce a 3 x (2*k-1) matrix.

    Parameters
    ----------
    gt : np.ndarray
        Genotype matrix of shape (n*k*k, L), entries 0/1.
    n : int
        Number of individuals per lattice position.
    k : int
        Lattice side length.

    Returns
    -------
    np.ndarray
        A (3, 2*k-1) array: rows correspond to DAF bins (rare, low, common), columns to
        Manhattan distances 0..(2k-2) (indexed as 1..2k-1 in the R code; here 0-based).
    """
    gt = np.asarray(gt)
    if gt.ndim != 2:
        raise ValueError("gt must be a 2D array of shape (n*k*k, L)")
    n_gt, n_lc = gt.shape
    if n_gt != n * k * k:
        raise ValueError("Genotype matrix is the wrong size: expected n*k*k rows")

    max_dist = 2 * k
    results = np.full((n_lc, max_dist - 1), np.nan, dtype=float)

    a_f = gt.mean(axis=0)  # allele frequencies per locus
    # Frequency bin: 1=rare (<0.04), 2=low (<0.1), 3=common (>=0.1)
    a_b = np.where(a_f < 0.04, 1, np.where(a_f < 0.1, 2, 3))

    d_m = sample_distance_matrix(n, k) + 1  # add one for 1-based indexing as in R

    for m in range(n_lc):
        this_shared = np.zeros(max_dist - 1, dtype=float)
        this_all = np.zeros(max_dist - 1, dtype=float)

        col = gt[:, m].astype(bool)
        # Iterate pairs i<j; replicate R logic (only count pairs where i has allele)
        for i in range(n_gt - 1):
            if col[i]:
                # distances from i to all j>i
                d_slice = d_m[i, i + 1 :]
                # increment all potential pairs for this i
                for d in d_slice:
                    this_all[d - 1] += 1
                # among them, increment shared where j also has the allele
                if col[i + 1 :].any():
                    js = np.nonzero(col[i + 1 :])[0]
                    # js are relative indices; map distances
                    for rel_j in js:
                        d = d_slice[rel_j]
                        this_shared[d - 1] += 1

        with np.errstate(invalid="ignore", divide="ignore"):
            frac = this_shared / this_all
        results[m, :] = frac

    # Normalize by allele frequency per locus (broadcasting)
    with np.errstate(invalid="ignore", divide="ignore"):
        results = results / a_f[:, None]

    # Summarize by frequency bin (rows: rare, low, common)
    summary = np.zeros((3, max_dist - 1), dtype=float)
    for i_bin in (1, 2, 3):
        mask = a_b == i_bin
        if mask.any():
            summary[i_bin - 1, :] = np.nanmean(results[mask, :], axis=0)
        else:
            summary[i_bin - 1, :] = np.nan

    return summary


def plot_excess_allele_sharing(
    sr: np.ndarray,
    labels: Tuple[str, str, str] = ("<0.04", "0.04-0.1", ">0.1"),
    log_scale: bool = False,
    **kwargs,
) -> None:
    """Plot the excess allele sharing curves.

    Parameters
    ----------
    sr : np.ndarray
        Matrix with shape (R, D) where R is number of frequency bins (typically 3)
        and D is the number of distance bins (2*k-1).
    labels : tuple of str
        Legend labels for each row.
    log_scale : bool
        If True, plot log10 values.
    **kwargs : dict
        Passed through to `plt.plot` for the first line (e.g., linewidth).
    """
    sr = np.asarray(sr)
    if sr.shape[0] != len(labels):
        raise ValueError("Please supply labels of the right length")

    y = np.log10(sr) if log_scale else sr

    x = np.arange(1, y.shape[1] + 1)
    plt.plot(x, y[0, :], **kwargs)
    for i in range(1, y.shape[0]):
        plt.plot(x, y[i, :])

    plt.xlabel("Distance")
    plt.ylabel("lo$_{10}$(Excess allele sharing)" if log_scale else "Excess allele sharing")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Reference line at 0 (log-scale) or 1 (linear), matching the R code's abline(h=1-log.scale)
    ref = 0.0 if log_scale else 1.0
    plt.axhline(ref, linestyle="--", linewidth=3)

    plt.legend(labels, title="DAF", frameon=False, loc="upper right")
    plt.tight_layout()
