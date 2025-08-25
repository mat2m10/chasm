
# demo_generate_admixed_genotypes.py
import argparse
import numpy as np
import pandas as pd
from admix_sim import simulate_admixed_genotypes, sample_ancestry_proportions

def main():
    ap = argparse.ArgumentParser(description="Generate an admixed genotype matrix (CSV).")
    ap.add_argument("--n_ind", type=int, default=500, help="Number of individuals")
    ap.add_argument("--n_snps", type=int, default=10000, help="Number of SNPs")
    ap.add_argument("--K", type=int, default=2, help="Number of source populations")
    ap.add_argument("--Fst", type=float, default=0.05, help="Drift (Balding–Nichols)")
    ap.add_argument("--seed", type=int, default=1, help="Random seed")
    ap.add_argument("--out", type=str, default="admixed_genotypes.csv", help="Output CSV path")
    ap.add_argument("--centers", type=str, default="", help="Comma-separated centers (e.g., '0.7,0.3') for ancestry proportions")
    ap.add_argument("--conc", type=float, default=100.0, help="Dirichlet concentration around centers")

    args = ap.parse_args()

    if args.centers:
        centers = np.array([float(x) for x in args.centers.split(",")], dtype=float)
        if centers.size != args.K:
            raise SystemExit("centers must have length K")
        W = sample_ancestry_proportions(args.n_ind, args.K, centers=centers, conc=args.conc, seed=args.seed)
    else:
        W = None

    G = simulate_admixed_genotypes(args.n_ind, args.n_snps, K=args.K, Fst=args.Fst, W=W, seed=args.seed)

    # Save as CSV (individuals as rows, SNPs as columns)
    df = pd.DataFrame(G, columns=[f"SNP{j+1}" for j in range(G.shape[1])])
    df.insert(0, "IID", [f"ind{i+1}" for i in range(G.shape[0])])
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {G.shape}")

if __name__ == "__main__":
    main()
