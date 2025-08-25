"""

This script simulates genotype presence/absence across a k x k lattice of demes
with n individuals per deme, under a simple spatial diffusion model meant to
approximate isolation-by-distance. It then computes excess allele sharing by
lattice distance and writes both a CSV and a PDF matching the R script outputs.

Outputs
-------
- CSV:  Allele_sharing_M_<M>_k_<k>_N_<n_loci*n_mutations>.new.txt
- PDF:  Allele_sharing_M_<M>_k_<k>_N_<n_loci*n_mutations>.pdf

Notes
-----
The core API mirrors the R:
- simulate_genotypes(n_loci, n_mutations, n, k, M) -> (n*k*k, n_loci*n_mutations) 0/1 array
- allele_sharing_by_distance(gt, n, k) -> (3, 2*k-1) array

The simulator uses a Gaussian kernel centered at a random lattice location per
mutation; the spatial scale increases with migration-rate-like parameter M.
"""
from __future__ import annotations

import math
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from allele_sharing_lib import allele_sharing_by_distance, plot_excess_allele_sharing


def _grid_coords(k: int) -> np.ndarray:
    """Return an array of shape (k*k, 2) with integer grid coordinates."""
    xs, ys = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
    return coords  # (k*k, 2)


def _gaussian_field(center: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Compute a normalized Gaussian field over the k x k lattice."""
    coords = _grid_coords(k)
    d2 = np.sum((coords - center[None, :]) ** 2, axis=1)
    field = np.exp(-d2 / (2.0 * (sigma ** 2)))  # (k*k,)
    # Normalize peak to 1
    if field.max() > 0:
        field = field / field.max()
    return field.reshape(k, k)


def simulate_genotypes(
    n_loci: int,
    n_mutations: int,
    n: int,
    k: int,
    M: float,
    seed: int | None = 1,
) -> np.ndarray:
    """Simulate a genotype (presence/absence) matrix under spatial clustering.

    Parameters
    ----------
    n_loci : int
        Number of coalescent replicates (conceptual).
    n_mutations : int
        Number of variants per locus to draw.
    n : int
        Number of individuals sampled per deme.
    k : int
        Lattice side length (k x k demes).
    M : float
        Migration-rate-like parameter; higher values increase spatial spread.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Binary matrix with shape (n*k*k, n_loci*n_mutations).
    """
    rng = np.random.default_rng(seed)

    nkk = n * k * k
    total_vars = n_loci * n_mutations
    gt = np.zeros((nkk, total_vars), dtype=np.uint8)

    # Precompute per-deme individual indices (rows in gt)
    # deme index order: (x, y) rasterized with x major order, matching _grid_coords
    per_deme_rows = []
    for x in range(k):
        for y in range(k):
            start = (x * k + y) * n
            rows = np.arange(start, start + n, dtype=int)
            per_deme_rows.append(rows)
    per_deme_rows = np.array(per_deme_rows, dtype=object)  # length k*k of arrays of length n

    # Map M -> spatial scale (sigma). Ensure sigma >= 0.75 and grows with M.
    sigma = 0.75 + 0.35 * k * (M / (1.0 + M))

    var_idx = 0
    for _ in range(n_loci):
        for _ in range(n_mutations):
            # Random center on the lattice
            cx = rng.integers(0, k)
            cy = rng.integers(0, k)

            field = _gaussian_field(np.array([cx, cy]), k, sigma)  # peak=1

            # Draw a global scale (target frequency) skewed to rare variants
            # Beta(a,b) with a<<b yields mostly small values
            scale = rng.beta(0.3, 10.0)
            # Cap the max probability to avoid fixation-like events
            max_p = min(0.6, scale * 2.0)
            p = np.clip(scale * field, 1e-6, max_p)

            # Sample n individuals per deme
            for deme_idx in range(k * k):
                rows = per_deme_rows[deme_idx]
                gt[rows, var_idx] = rng.binomial(1, p.ravel()[deme_idx], size=n)

            var_idx += 1

    return gt


def main():
    parser = argparse.ArgumentParser(description="Allele sharing simulation")
    parser.add_argument("--k", type=int, default=20, help="Lattice side length (k x k demes)")
    parser.add_argument("--n", type=int, default=2, help="Samples per deme")
    parser.add_argument("--M", type=float, default=10.0, help="Migration-like rate controlling spread")
    parser.add_argument("--n_loci", type=int, default=1000, help="Number of loci (replicates)")
    parser.add_argument("--n_mutations", type=int, default=1, help="Variants per locus")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")

    args = parser.parse_args()
    k = args.k
    n = args.n
    M = args.M
    n_loci = args.n_loci
    n_mut = args.n_mutations

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Simulate genotypes
    gt = simulate_genotypes(n_loci, n_mut, n, k, M, seed=args.seed)

    # Compute allele sharing by distance
    asd = allele_sharing_by_distance(gt, n, k)

    # Save CSV (comma-separated, no row/col names)
    N = n_loci * n_mut
    stem = f"Allele_sharing_M_{M}_k_{k}_N_{N}"
    csv_path = outdir / f"{stem}.new.txt"
    np.savetxt(csv_path, asd, delimiter=",", fmt="%.6g")

    # Plot PDF (log10 scale), with limits roughly matching R example
    pdf_path = outdir / f"{stem}.pdf"
    plt.figure()
    plot_excess_allele_sharing(asd, labels=("<0.04", "0.04-0.1", ">0.1"), log_scale=True)
    # Match ylim=c(-2,2), xlim=c(0, 2*k) as in R (note Python is 1..2k-1 on x)
    plt.ylim(-2, 2)
    plt.xlim(0, 2 * k)
    # Thicker lines & larger fonts akin to par(lwd=4), cex=1.4
    for line in plt.gca().get_lines():
        line.set_linewidth(4)
    plt.gcf().set_size_inches(8, 5)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
