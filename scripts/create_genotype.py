import numpy as np
import pandas as pd
import subprocess
import os

# --- Compile kpop.so if needed ---
def compile_kpop_so():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chasm/rare_var_simulation/rstudio_rare_var_simulation"))
    src_file = os.path.join(base_dir, "kpop.c")
    out_file = os.path.join(base_dir, "kpop.so")
    
    if not os.path.isfile(out_file):
        print("Compiling kpop.c into kpop.so using gcc...")
        compile_cmd = f"gcc -shared -fPIC -o {out_file} {src_file}"
        result = subprocess.run(compile_cmd, shell=True, cwd=base_dir, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error compiling kpop.c:")
            print(result.stderr)
            raise RuntimeError("C compilation failed.")

# --- Run the R genotype simulation ---
def run_r_simulation(G, L, c, k, M):
    compile_kpop_so()

    r_script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../chasm/rare_var_simulation/rstudio_rare_var_simulation/create_geno.R")
    )
    working_dir = os.path.dirname(r_script_path)

    commands = [
        f"G <- {G}",
        f"L <- {L}",
        f"c <- {c}",
        f"k <- {k}",
        f"M <- {M}",
        f"setwd('{working_dir}')",
        f"source('create_geno.R', echo=TRUE)"
    ]
    
    r_script = ";".join(commands)
    result = subprocess.run(['Rscript', '-e', r_script], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error executing R script:")
        print(result.stderr)
        raise RuntimeError("R simulation failed.")

# --- Utility helpers ---
def get_simulated_file_path(G, L, c, k, M, path):
    raw_path = os.path.join(path, "raw")
    filename = f"simulated_genotypes_G{G}_L{L}_c{c}_k{k}_M{M}.csv"
    return os.path.join(raw_path, filename)

def move_output_file(G, L, c, k, M, path):
    os.makedirs(os.path.join(path, "raw"), exist_ok=True)
    filename = f"simulated_genotypes_G{G}_L{L}_c{c}_k{k}_M{M}.csv"
    os.system(f"mv {filename} {os.path.join(path, 'raw')}")

def summarize_genotypes(df):
    return pd.DataFrame({
        f'G{i//2 + 1}': np.where(df.iloc[:, i-1] + df.iloc[:, i] == 2, 2, df.iloc[:, i-1] + df.iloc[:, i])
        for i in range(1, df.shape[1], 2)
    })

# --- Main simulation functions ---
def simulate_genos(G, L, c, k, M, F, path):
    run_r_simulation(G, L, c, k, M)
    move_output_file(G, L, c, k, M, path)

    path_sim_file = get_simulated_file_path(G, L, c, k, M, path)
    df = pd.read_csv(path_sim_file)
    os.remove(path_sim_file)

    genotypes = summarize_genotypes(df)
    genotypes = genotypes.loc[:, genotypes.nunique() > 1]

    genotypes["populations"] = [i+1 for i in range(k*k) for _ in range(c)]

    dfs = []
    required = {0, 1, 2}

    for pop in sorted(genotypes["populations"].unique()):
        pop_df = genotypes[genotypes["populations"] == pop].drop(columns="populations")
        
        for col in pop_df.columns:
            if not required.issubset(set(pop_df[col])):
                idx = np.random.choice(pop_df.index, 3, replace=False)
                pop_df.loc[idx[0], col] = 0
                pop_df.loc[idx[1], col] = 1
                pop_df.loc[idx[2], col] = 2

            val_counts = pop_df[col].value_counts().reindex([0, 1, 2], fill_value=0)
            total = val_counts.sum()
            q = (2 * val_counts[2] + val_counts[1]) / (2 * total)
            q = min(q, 1 - q)
            p = 1 - q

            f_maj = max((p**2) + F*p*q, 0.001)
            f_het = max((2*p*q)*(1-F), 0.001)
            f_min = max((q**2) + F*p*q, 0.001)

            total_f = f_maj + f_het + f_min
            error = (total_f - 1) / 3
            f_maj += error; f_het += error; f_min += error

            f_sum = f_maj + f_het + f_min
            f_maj /= f_sum; f_het /= f_sum; f_min /= f_sum

            pop_df[col] = np.random.choice([1.0, 0.0, -1.0], size=total, p=[f_maj, f_het, f_min])

        dfs.append(pop_df)

    genotype_data = pd.concat(dfs, ignore_index=True)
    genotype_data.to_pickle(os.path.join(path, "raw", "geno_simul.pkl"))
    return genotype_data

def simulate_genos_raw(G, L, c, k, M, path):
    run_r_simulation(G, L, c, k, M)
    move_output_file(G, L, c, k, M, path)
    path_sim_file = get_simulated_file_path(G, L, c, k, M, path)
    df1 = pd.read_csv(path_sim_file)

    run_r_simulation(G, L, c, k, M)
    move_output_file(G, L, c, k, M, path)
    df2 = pd.read_csv(path_sim_file)

    os.remove(path_sim_file)

    summed = df1 + df2
    summed.to_pickle(os.path.join(path, "raw", "geno_simul.pkl"))
    return summed
