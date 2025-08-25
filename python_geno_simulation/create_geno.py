import numpy as np
import pandas as pd
import subprocess
import os

# Example values (replace with sys.argv or function args if needed)
G = 5
L = 10
c = 2
k = 3
M = 100

print(f"G: {G}")
print(f"L: {L}")
print(f"c: {c}")
print(f"k: {k}")
print(f"M: {M}")

# Example of using numpy to replicate the 'rep' function in R
example_rep = np.tile(np.arange(1, G + 1), k)  # repeat sequence 1:G, times = k
print(example_rep)

# ---- Load external functionality ----
# In R you had: source("association.R") and dyn.load("kpop.so")
# In Python, you’d import modules or use ctypes to load shared libraries.
# Example (if you have equivalent Python or compiled library):
# from association import simulate_genotypes
# or with ctypes:
# from ctypes import CDLL
# lib = CDLL("./kpop.so")

# ---- Define a simulate function ----
def simulate_genotypes(G, L, c, k, M):
    """
    Placeholder simulation for genotypes.
    Replace with real logic or wrap C/C++ library.
    """
    # Simulate genotypes as random integers (0,1,2)
    return np.random.randint(0, 3, size=(M, L))

# ---- Save as CSV ----
def simulate_and_save_csv(G, L, c, k, M):
    gt = simulate_genotypes(G, L, c, k, M)

    # Convert to DataFrame
    df = pd.DataFrame(gt)

    # Construct file name with variable values
    file_name = f"simulated_genotypes_G{G}_L{L}_c{c}_k{k}_M{M}.csv"

    # Save as CSV
    df.to_csv(file_name, index=False)

    print(f"Simulated genotypes saved as: {file_name}")

# ---- Run simulation ----
simulate_and_save_csv(G, L, c, k, M)
