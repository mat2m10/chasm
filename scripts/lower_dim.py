import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def compute_pcs(
    df: pd.DataFrame,
    *,
    min_components: int = 4,
    max_components: int = 50,
    prefix: str = "PC",
    fallback_components: int = 4,
) -> pd.DataFrame:
    """
    Scale columns and run PCA with safe component bounds.
    Falls back to a zero matrix with `fallback_components` PCs if anything fails.
    """
    try:
        if df is None or df.shape[0] == 0:
            raise ValueError("Input has no rows.")
        if df.shape[1] == 0:
            raise ValueError("Input has no columns.")

        # Bound by both features and samples to avoid sklearn errors
        n_features = df.shape[1]
        n_samples = df.shape[0]
        n_components = max(min(n_features, n_samples, max_components), min_components)

        scaled = StandardScaler().fit_transform(df)
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(scaled)

        labels = [f"{prefix}{i}" for i in range(1, n_components + 1)]
        return pd.DataFrame(pcs, index=df.index, columns=labels)

    except Exception as e:
        # Print once, return deterministic fallback shape
        print(f"[{prefix}] PCA fallback used:", e)
        labels = [f"{prefix}{i}" for i in range(1, fallback_components + 1)]
        return pd.DataFrame(
            np.zeros((0 if df is None else len(df), fallback_components)),
            index=(None if df is None else df.index),
            columns=labels,
        )

def attach_metadata(pcs: pd.DataFrame, meta: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in meta.columns]
    if missing:
        raise KeyError(f"Metadata missing required columns: {missing}")
    # Use align/join to respect current index; no SettingWithCopy issues
    return pcs.join(meta.loc[pcs.index, cols])