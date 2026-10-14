import numpy as np


def make_pheno(humans):
    pheno = humans[["x", "y", "populations"]].copy()
    pheno["no_bias"] = humans["z_outbred"]
    
    k = int(np.sqrt(len(humans["populations"].unique())))
    
    # --- existing ---
    pheno["linear"] = pheno["x"] + pheno["y"]

    freq_x, freq_y = 3, 2
    sx = np.sin(pheno["x"] * freq_x * np.pi / k)
    sy = np.sin(pheno["y"] * freq_y * np.pi / k)
    pheno["sine_x_mix"]   = np.round(sx, 2)
    pheno["sine_y_mix"]   = np.round(sy, 2)
    pheno["sine_x_y_mix"] = np.round(sx + sy, 2)

    n = int(k - k // 3)
    pheno["discrete"] = ((pheno["x"] == n) & (pheno["y"] == n)).astype(int)

    # --- new: hard for PCA ---

    # 1. Multiplicative interaction — requires PC1*PC2, invisible to additive PCs
    pheno["interaction"] = np.round(sx * sy, 2)

    # 2. Radial pattern — distance from grid center, rotationally symmetric
    #    PCA axes are Cartesian, so radial structure has no preferred axis
    cx, cy = k / 2, k / 2
    r = np.sqrt((pheno["x"] - cx)**2 + (pheno["y"] - cy)**2)
    pheno["radial"] = np.round(np.sin(r * np.pi / (k / 2)), 2)

    # 3. High-frequency checkerboard — aliased, variance spreads across many PCs
    pheno["checkerboard"] = (((pheno["x"] + pheno["y"]) % 2) * 2 - 1).astype(int)


    # 4. XOR-like discrete — quadrant sign flip, zero row/column marginals
    #    mean per row = 0, mean per column = 0, signal is pure interaction
    pheno["xor_quadrant"] = np.sign(
        (pheno["x"] - cx) * (pheno["y"] - cy)
    ).astype(int)

    # 5. Spiral — continuous but neither axis-aligned nor radially symmetric
    #    requires many PCs and their interactions to approximate
    theta = np.arctan2(pheno["y"] - cy, pheno["x"] - cx)
    pheno["spiral"] = np.round(np.sin(r + theta), 2)

    # 6. Random smooth (RBF blobs) — spatially correlated but not structured
    #    mimics realistic confounding from isolation-by-distance
    rng = np.random.default_rng(42)
    n_centers = 5
    centers = rng.uniform(0, k, (n_centers, 2))
    blob = np.zeros(len(pheno))
    for cx_, cy_ in centers:
        d2 = (pheno["x"] - cx_)**2 + (pheno["y"] - cy_)**2
        blob += np.exp(-d2 / (k / 3)**2)
    pheno["rbf_blobs"] = np.round((blob - blob.mean()) / blob.std(), 2)

    return pheno