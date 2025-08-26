import pandas as pd
import numpy as np
def attach_populations(n_rows: int, k: int, c: int) -> pd.DataFrame:
    """Construct a populations DataFrame with balanced groups.
    Pop IDs go from 1..k*k, each repeated c times.
    """
    num_pops = k * k
    expected_rows = num_pops * c
    if n_rows != expected_rows:
        raise ValueError(
            f"Row count {n_rows} != expected k*k*c = {expected_rows}. Check inputs."
        )
    pops = np.repeat(np.arange(1, num_pops + 1), c)
    return pd.DataFrame({"populations": pops})


def compute_human_metadata(populations: pd.Series, k: int) -> pd.DataFrame:
    """Compute per-individual metadata (grid coordinates, scaled population index).
    Adds x, y (1-based grid), z (0 placeholder), and population_fraction.
    """
    pops = populations.astype(int)
    x = ((pops - 1) % k) + 1
    y = ((pops - 1) // k) + 1
    z = np.zeros_like(x)
    population_fraction = pops / (k * k)
    return pd.DataFrame({
        "populations": pops,
        "x": x,
        "y": y,
        "z": z,
        "population": population_fraction,
    })

def add_rgb_from_xyz(df_xyz: pd.DataFrame) -> pd.DataFrame:
    """Add normalized RGB columns from x,y,z in a vectorized manner (safe for zeros)."""
    x = df_xyz["x"].to_numpy()
    y = df_xyz["y"].to_numpy()
    z = df_xyz["z"].to_numpy()
    r_den = np.max(x) if np.max(x) != 0 else 1
    g_den = np.max(y) if np.max(y) != 0 else 1
    b_den = np.max(z) if np.max(z) != 0 else 1
    return df_xyz.assign(
        r = x / r_den,
        g = y / g_den,
        b = z / b_den,
    )