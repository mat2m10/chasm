import numpy as np
import pandas as pd

from chasm.pca import get_n_pcs  # re-exported for backwards compatibility

def PC_distance_to_snp(geno_std, snps_pcs, chosen_snp, n_pcs=25):
      # index = SNPs, columns = PCs

    X = snps_pcs.to_numpy(dtype=float)
    i = snps_pcs.index.get_loc(chosen_snp)

    diff = X - X[i]
    dists = np.sqrt(np.sum(diff * diff, axis=1))

    dist_df = pd.DataFrame({"snp": snps_pcs.index, "distance": dists})
    dist_df = dist_df.sort_values("distance", ascending=False)

    dmin, dmax = dist_df["distance"].min(), dist_df["distance"].max()
    dist_df["proximity"] = 1.0 - (dist_df["distance"] - dmin) / (dmax - dmin)
    dist_df = dist_df[dist_df['snp']!=chosen_snp]

    return dist_df

def pearson_distance_to_snp(X, cols, chosen_snp):

    i = cols.get_loc(chosen_snp)

    x = X[:, i]
    n = X.shape[0]

    # correlation of chosen SNP with every SNP (since standardized)
    r = (X.T @ x) / (n - 1)                            # (n_snps,)

    dist = 1.0 - r                                     # Pearson distance

    dist_df = pd.DataFrame({"snp": cols, "distance": dist})
    dist_df = dist_df.sort_values("distance", ascending=False)

    dmin, dmax = dist_df["distance"].min(), dist_df["distance"].max()
    dist_df["proximity"] = 1.0 - (dist_df["distance"] - dmin) / (dmax - dmin)
    dist_df = dist_df[dist_df['snp']!=chosen_snp]
    return dist_df

def marginal_estimation(df1,df2):
    merged = (
        df2
        .merge(
            df1[['snp', 'proximity']],
            left_on='names',
            right_on='snp',
            how='inner'
        )
    )
    result = (merged['estimated_betas_std'] * merged['proximity']).sum()
    return result
