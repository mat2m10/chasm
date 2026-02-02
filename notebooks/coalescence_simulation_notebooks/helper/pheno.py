import numpy as np


def make_pheno(humans):
    pheno = humans[["x", "y", "populations"]].copy()
    pheno["no_bias"] = humans["z_outbred"]
    pheno["linear"] = pheno["x"] + pheno["y"]

    k = int(np.sqrt(len(humans["populations"].unique())))

    freq_x = 3
    freq_y = 2

    pheno["sine_x_mix"] = np.round(np.sin(pheno["x"] * freq_x * np.pi / k), 2)
    pheno["sine_y_mix"] = np.round(np.sin(pheno["y"] * freq_y * np.pi / k), 2)
    pheno["sine_x_y_mix"] = np.round(pheno["sine_x_mix"] + pheno["sine_y_mix"], 2)

    n = int(k - k // 3)
    pheno["discrete"] = ((pheno["x"] == n) & (pheno["y"] == n)).astype(int)
    return pheno
