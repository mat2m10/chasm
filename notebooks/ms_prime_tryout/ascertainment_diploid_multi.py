# %%writefile ascertainment_diploid_multi.py
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Literal

import numpy as np
import pandas as pd


@dataclass
class AscertainmentConfig:
    # Backward-compatible single-pop option (used if discovery_pops is None)
    discovery_pop: str = "A"
    # Allow selecting multiple populations
    discovery_pops: Optional[Sequence[str]] = None
    # How to evaluate the MAF threshold across multiple pops
    #  - "pooled": pool chosen pops (optionally weighted) and test pooled MAF (default)
    #  - "any": keep SNP if it passes MAF in >=1 chosen pop
    #  - "all": keep SNP if it passes MAF in every chosen pop
    maf_mode: Literal["pooled", "any", "all"] = "pooled"
    # Optional weights for pooling by population label (e.g., {"EUR":2, "AFR":1})
    discovery_weights: Optional[Dict[str, float]] = None

    maf_min: float = 0.05
    maf_max: float = 0.50
    n_snps_target: int = 600_000
    ld_r2: float = 0.2
    ld_window_bp: int = 50_000
    ld_step_bp: int = 5_000
    random_seed: int = 1


def _calc_maf_from_genotypes(g: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Diploid genotypes 0/1/2 → MAF in [0, 0.5]. Optional per-individual weights."""
    if g.size == 0:
        return np.nan
    if weights is None:
        p = g.mean() / 2.0
    else:
        w = np.asarray(weights, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if w.sum() == 0:
            return np.nan
        p = float((g @ w) / (2.0 * w.sum()))
    p = float(np.clip(p, 0, 1))
    return float(min(p, 1 - p))


def _collapse_haplotypes_to_diploid(G_hap: np.ndarray, pairing: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """
    Collapse haplotype 0/1 matrix (variants x haplotypes) to diploid 0/1/2 (variants x individuals).
    pairing: optional list of (hap1_col, hap2_col). If None, assume consecutive pairs.
    """
    n_vars, n_haps = G_hap.shape
    if pairing is None:
        if n_haps % 2 != 0:
            raise ValueError("Number of haplotypes is odd; provide explicit pairing.")
        G_dip = G_hap.reshape(n_vars, n_haps // 2, 2).sum(axis=2)
        return G_dip.astype(np.int8)

    # explicit pairing
    n_ind = len(pairing)
    G_dip = np.empty((n_vars, n_ind), dtype=np.int16)
    for i, (h1, h2) in enumerate(pairing):
        G_dip[:, i] = G_hap[:, h1] + G_hap[:, h2]
    return G_dip.astype(np.int8)


def _windowed_ld_prune(G: np.ndarray, positions: np.ndarray, r2_thresh: float, window: int, step: int) -> np.ndarray:
    """
    Simple LD pruning by r^2 in bp windows.
    Returns indices of kept variants relative to input.
    """
    keep = []
    last_kept_pos = -10**18
    n_vars = G.shape[0]
    for i in range(n_vars):
        if positions[i] < last_kept_pos + step:
            continue
        ok = True
        # check previous kept variants within the window
        for j in reversed(keep):
            if positions[i] - positions[j] > window:
                break
            gi = G[i]; gj = G[j]
            ci = gi - gi.mean()
            cj = gj - gj.mean()
            denom = float((ci @ ci) * (cj @ cj))
            if denom <= 0:
                continue
            r = float((ci @ cj) / np.sqrt(denom))
            if r * r > r2_thresh:
                ok = False; break
        if ok:
            keep.append(i)
            last_kept_pos = positions[i]
    return np.array(keep, dtype=int)


def _get_discovery_mask(samples_indiv: pd.DataFrame, discovery_sample_col: str, config: AscertainmentConfig) -> np.ndarray:
    if config.discovery_pops is None:
        mask = (samples_indiv[discovery_sample_col].values == config.discovery_pop)
    else:
        mask = samples_indiv[discovery_sample_col].isin(config.discovery_pops).values
    return mask


def _get_weights_for_mask(samples_indiv: pd.DataFrame, discovery_sample_col: str, mask: np.ndarray, config: AscertainmentConfig) -> Optional[np.ndarray]:
    """Return per-individual weights (length = n_individuals_in_mask) or None for unweighted."""
    if config.maf_mode != "pooled":
        return None  # weights only apply to pooled mode
    if not config.discovery_weights:
        return None
    labels = samples_indiv.loc[mask, discovery_sample_col].values
    w = np.array([config.discovery_weights.get(lbl, 0.0) for lbl in labels], dtype=float)
    if np.all(w == 0):
        return None
    return w


def ascertain_chip_diploid(
    genotypes_hap_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    config: AscertainmentConfig = AscertainmentConfig(),
    pairing: Optional[Sequence[Tuple[int, int]]] = None,
    discovery_sample_col: str = "population",
    outdir: Path = Path("ascertained_chip")
) -> dict:
    """
    Perform array-like ascertainment and return diploid genotypes.

    Inputs:
      - genotypes_hap_df: variants x haplotypes (0/1)
      - sites_df: must include a 'pos' column (int bp)
      - samples_df: must include 'sample_index' (haplotype index) and discovery_sample_col (e.g., 'population')
      - pairing: optional explicit list of (hap1, hap2); otherwise pairs consecutive haplotypes
      - config: AscertainmentConfig with thresholds & seeds

    Outputs (also saved to outdir):
      - genotypes_array_diploid.parquet: variants x individuals (0/1/2)
      - sites_array.parquet: filtered sites
      - samples_individuals.parquet: one row per individual (with discovery_sample_col if available)
      - manifest.json: parameters & counts
      - ascertainment_index.parquet: indices of kept variants relative to original
    """
    rng = np.random.default_rng(config.random_seed)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Collapse to diploid
    G_hap = genotypes_hap_df.values  # (V x H)
    G_dip = _collapse_haplotypes_to_diploid(G_hap, pairing=pairing)  # (V x I)
    n_vars, n_ind = G_dip.shape

    # 2) Build per-individual sample table
    if pairing is None:
        # consecutive pairs -> individual_id = 0..n_ind-1
        samples_indiv = pd.DataFrame({"individual_id": np.arange(n_ind)})
        if discovery_sample_col in samples_df.columns:
            pops = []
            for i in range(n_ind):
                hap0 = 2*i
                if hap0 < len(samples_df):
                    pops.append(samples_df.iloc[hap0][discovery_sample_col])
                else:
                    pops.append(None)
            samples_indiv[discovery_sample_col] = pops
    else:
        pops = []
        for h1, h2 in pairing:
            if discovery_sample_col in samples_df.columns:
                p1 = samples_df.iloc[h1][discovery_sample_col]
                p2 = samples_df.iloc[h2][discovery_sample_col]
                pops.append(p1 if p1 == p2 else p1)
            else:
                pops.append(None)
        samples_indiv = pd.DataFrame({"individual_id": np.arange(n_ind), discovery_sample_col: pops})

    # 3) Discovery panel mask
    if discovery_sample_col not in samples_indiv.columns:
        raise ValueError(f"'{discovery_sample_col}' not found in individuals table; cannot define discovery panel.")
    disc_mask = _get_discovery_mask(samples_indiv, discovery_sample_col, config)
    if disc_mask.sum() == 0:
        target = config.discovery_pops if config.discovery_pops is not None else config.discovery_pop
        raise ValueError(f"No individuals found for discovery populations {target!r}.")

    # 4) Compute MAF based on mode
    G_disc = G_dip[:, disc_mask]  # (V x I_disc)

    if config.maf_mode == "pooled":
        weights = _get_weights_for_mask(samples_indiv, discovery_sample_col, disc_mask, config)
        maf_vals = np.apply_along_axis(_calc_maf_from_genotypes, 1, G_disc, weights)
        maf_keep = (maf_vals >= config.maf_min) & (maf_vals <= config.maf_max)

    else:
        # compute per-pop MAFs among the chosen discovery pops
        if config.discovery_pops is None:
            chosen = [config.discovery_pop]
        else:
            chosen = list(config.discovery_pops)

        maf_by_pop = []
        for pop in chosen:
            pop_mask = (samples_indiv[discovery_sample_col].values == pop) & disc_mask
            G_pop = G_dip[:, pop_mask]
            maf_pop = np.apply_along_axis(_calc_maf_from_genotypes, 1, G_pop)
            maf_by_pop.append(maf_pop)

        maf_by_pop = np.vstack(maf_by_pop)  # (P x V)
        within = (maf_by_pop >= config.maf_min) & (maf_by_pop <= config.maf_max)  # (P x V)
        if config.maf_mode == "any":
            maf_keep = within.any(axis=0)
        elif config.maf_mode == "all":
            maf_keep = within.all(axis=0)
        else:
            raise ValueError(f"Unknown maf_mode: {config.maf_mode}")

    G_maf = G_dip[maf_keep]
    sites_maf = sites_df.loc[maf_keep].reset_index(drop=True)

    # 5) LD prune (windowed r^2)
    keep_idx_local = _windowed_ld_prune(
        G_maf, sites_maf["pos"].to_numpy().astype(int),
        r2_thresh=config.ld_r2, window=config.ld_window_bp, step=config.ld_step_bp
    )

    G_pruned = G_maf[keep_idx_local]
    sites_pruned = sites_maf.iloc[keep_idx_local].reset_index(drop=True)

    # 6) Downsample to target SNP count
    if G_pruned.shape[0] > config.n_snps_target:
        rng_idx = rng.choice(G_pruned.shape[0], size=config.n_snps_target, replace=False)
        rng_idx.sort()
        G_chip = G_pruned[rng_idx]
        sites_chip = sites_pruned.iloc[rng_idx].reset_index(drop=True)
        idx_chip = np.flatnonzero(maf_keep)[keep_idx_local][rng_idx]
    else:
        G_chip = G_pruned
        sites_chip = sites_pruned
        idx_chip = np.flatnonzero(maf_keep)[keep_idx_local]

    # 7) Save outputs
    G_chip_df = pd.DataFrame(G_chip)  # variants x individuals
    sites_chip.to_parquet(outdir / "sites_array.parquet", index=False)
    G_chip_df.to_parquet(outdir / "genotypes_array_diploid.parquet", index=False)
    samples_indiv.to_parquet(outdir / "samples_individuals.parquet", index=False)
    pd.DataFrame({"ascertainment_index": idx_chip}).to_parquet(outdir / "ascertainment_index.parquet", index=False)

    manifest = {
        "n_variants_in": int(n_vars),
        "n_individuals": int(n_ind),
        "n_discovery": int(disc_mask.sum()),
        "maf_mode": str(config.maf_mode),
        "discovery_pops": list(config.discovery_pops) if config.discovery_pops is not None else [config.discovery_pop],
        "n_after_maf": int(G_maf.shape[0]),
        "n_after_ld": int(G_pruned.shape[0]),
        "n_final": int(G_chip.shape[0]),
        "config": asdict(config),
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "G_dip_df": G_chip_df,
        "sites_chip": sites_chip,
        "samples_indiv": samples_indiv,
        "manifest": manifest,
        "ascertainment_index": idx_chip,
    }
