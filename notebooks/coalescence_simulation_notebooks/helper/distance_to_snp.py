import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_n_pcs(df: pd.DataFrame, n: int, scale: bool = True) -> pd.DataFrame:
    """
    Compute the first n principal components of a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (rows = samples, columns = features)
    n : int
        Number of principal components to return
    scale : bool, default=True
        Whether to standardize features before PCA

    Returns
    -------
    pd.DataFrame
        DataFrame containing the first n principal components
    """
    X = df.values

    if scale:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n)
    pcs = pca.fit_transform(X)

    pc_columns = [f"PC{i+1}" for i in range(n)]
    return pd.DataFrame(pcs, index=df.index, columns=pc_columns)

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