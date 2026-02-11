from pathlib import Path
import pandas as pd

from scripts.parse_vars import load_config


def load_data():
    PATH_VARS = Path("../../geno_simulation.txt")

    cfg = load_config(PATH_VARS)
    G = int(cfg["G"])
    L = int(cfg["L"])
    c = int(cfg["c"])
    k = int(cfg["k"])
    M = float(cfg["M"])

    F_outbred = 0.0

    path_pheno = f"simulation_data/G{G}_L{L}_c{c}_k{k}_M{M}_F{F_outbred}/phenotype/"
    humans = pd.read_pickle(f"{path_pheno}/humans.pkl")

    path_geno = f"simulation_data/G{G}_L{L}_c{c}_k{k}_M{M}_F{F_outbred}/genotype/"
    geno = (pd.read_pickle(f"{path_geno}/complete.pkl") + 1) / 2  # stays between 0 and 1
    return geno, humans
