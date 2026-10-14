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