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
git clone https://github.com/yourusername/chasm.git
cd chasm
poetry install
poetry shell
```