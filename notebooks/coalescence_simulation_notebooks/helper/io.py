import os
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

    # Override with CHASM_SIM_DATA_DIR; default is relative to the notebook folder.
    sim_root = os.environ.get("CHASM_SIM_DATA_DIR", "simulation_data")
    run_dir = f"{sim_root}/G{G}_L{L}_c{c}_k{k}_M{M}_F{F_outbred}"

    path_pheno = f"{run_dir}/phenotype/"
    humans = pd.read_pickle(f"{path_pheno}/humans.pkl")

    path_geno = f"{run_dir}/genotype/"
    geno = pd.read_pickle(f"{path_geno}/complete.pkl")
    return geno, humans