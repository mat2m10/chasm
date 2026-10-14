# chasm

**CHASM** (Correction and Heritability-Aware Simulation Model) is a simulation toolkit for generating and analyzing genotype-phenotype data under complex population structure, environmental influences, and heritability frameworks. It is designed for researchers exploring methodological biases in GWAS, burden testing, and rare variant inference under structured populations.

---

## ✨ Features

- 🧬 Simulation of admixed genotype data with customizable allele frequencies  
- 🌿 Modeling environmental covariates and gene-environment interactions  
- 🧪 Correction pipelines using ancestry factors and dimensionality reduction (UMAP, PCA, HDBSCAN)  
- 📊 Visualization of allele sharing, association statistics, and burden test results  
- 🔁 Integration with R-based coalescent models and rare variant association simulations  

---

## 🧰 Project Structure

```bash
chasm/
├── chasm/                         # Core Python package
├── scripts/                       # Utility scripts for genotype parsing, LD scoring, etc.
├── notebooks/                     # Jupyter notebooks for simulation pipelines
├── rstudio_geno_simulation/      # R and C code for allele sharing and burden testing
├── pyproject.toml                # Project configuration and dependencies (single source of truth)
├── poetry.lock                   # Locked dependency versions
├── README.md
```

## 🚀 Getting Started
```bash
git clone https://github.com/mat2m10/chasm.git
cd chasm
poetry install
poetry shell
```

### Configuring data paths

Data locations can be set with environment variables (or a `.env` file in
the repo root, read via `python-dotenv`). Unset variables fall back to the
previous defaults.

| Variable | Used by | Default |
|---|---|---|
| `CHASM_DATA_DIR` | `chasm.glob.glob.PATH_data` | `/mnt/d/chasm_data/` |
| `CHASM_SIM_DATA_DIR` | `helper/io.py` `load_data()` | `simulation_data` (relative to the notebook folder) |
| `CHASM_1000G_DIR` | `1000_genomes_notebooks/02 calculate PCs.ipynb` | `../../data/1000G` |
| `CHASM_PLINK2` | `1000_genomes_notebooks/02 calculate PCs.ipynb` | `../../plink2` |

### External tools (not tracked in git)

The PLINK binaries and the compiled kpop C code are no longer committed.
Install or build them locally:

- **PLINK 1.9 / PLINK 2**: download the Linux 64-bit builds from
  <https://www.cog-genomics.org/plink/> and <https://www.cog-genomics.org/plink/2.0/>,
  then place the `plink` and `plink2` executables in the repository root
  (the 1000 Genomes notebooks look for them there) or anywhere on your `PATH`:
  ```bash
  chmod +x plink plink2
  ```
- **kpop C extension** (used by the R scripts in `rstudio_geno_simulation/`):
  ```bash
  cd rstudio_geno_simulation
  R CMD SHLIB kpop.c   # produces kpop.o and kpop.so
  ```
