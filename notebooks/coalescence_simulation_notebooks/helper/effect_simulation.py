import numpy as np
from scipy import stats
import statsmodels.api as sm
import pandas as pd
import random

def fit_line(x_, y_, covs=None):
    x_ = np.asarray(x_).ravel()
    y_ = np.asarray(y_).ravel()

    # Original behavior if no covariates
    if covs is None:
        slope, intercept, r, p, se = stats.linregress(x_, y_)
        neglogp = -np.log10(p) if p > 0 else np.inf
        return slope, intercept, neglogp

    # With covariates: OLS on [x, covs] + intercept
    covs = np.asarray(covs)
    if covs.ndim == 1:
        covs = covs.reshape(-1, 1)

    X = np.column_stack([x_, covs])
    X = sm.add_constant(X, has_constant="add")  # adds intercept

    model = sm.OLS(y_, X, missing="drop").fit()

    slope = float(model.params[1])      # coefficient for x_
    intercept = float(model.params[0])  # intercept
    p = float(model.pvalues[1])         # p-value for x_ coefficient

    neglogp = -np.log10(p) if p > 0 else np.inf
    return slope, intercept, neglogp



def find_most_and_least_associated_snp(geno, pheno, most=4, least=3):
    snps = []
    slopes = []
    neglogps = []
    for snp in geno.columns:
        slope, intercept, neglogp = fit_line(geno[snp], pheno)
        snps.append(snp)
        slopes.append(slope)
        neglogps.append(neglogp)
    
    df = pd.DataFrame({
        "snp": snps,
        "slope": slopes,
        "neglogp": neglogps
    })
    df2 = df.sort_values("neglogp", ascending=False).reset_index(drop=True)
    best = df2.head(most)
    worst = df2.tail(least).sort_values("neglogp", ascending=True).reset_index(drop=True)
    out = pd.concat([best, worst], ignore_index=True)
    return best, worst

def simulate_effects(
    geno, 
    pheno,
    qty_best, 
    qty_worst, 
    percentage_polygenic_noise, 
    effect_chosen_snp, 
    effect_most_corr_snps, 
    effect_least_corr_snps, 
    effect_polygenic_noise):
    
    snp_names = list(geno.columns)
    best, worst = find_most_and_least_associated_snp(geno, pheno,qty_best,qty_worst)
    chosen_snp = best.loc[0].snp
    most_corr = []
    for i in range(qty_best-1):
        most_corr.append(best.loc[i+1].snp)
    least_corr = []
    for i in range(qty_worst):
        least_corr.append(worst.loc[i].snp)
    
    excluded = {chosen_snp}
    excluded.update(best["snp"].tolist())
    excluded.update(worst["snp"].tolist())
    # Pool to sample from
    eligible = [s for s in snp_names if s not in excluded]
    
    qty_poly_snps = int(((len(snp_names) - qty_best - qty_worst - 1) * percentage_polygenic_noise) / 100)
    
    # Safety: don't request more than available
    qty_poly_snps = min(qty_poly_snps, len(eligible))
    
    poly_snps = random.sample(eligible, qty_poly_snps)

    effect_most_corr_snps = effect_most_corr_snps/len(most_corr)
    effect_least_corr_snps = effect_least_corr_snps/len(least_corr)
    effect_polygenic_noise = effect_polygenic_noise/len(poly_snps)
    snps_meta_data = pd.DataFrame()
    snps_meta_data['names'] = snp_names
    snps_meta_data['categories'] = 'nul'
    snps_meta_data['raw_betas'] = 0.0

    mask = snps_meta_data["names"].isin(most_corr)
    snps_meta_data.loc[mask, "categories"] = "most correlated"
    snps_meta_data.loc[mask, "raw_betas"] = effect_most_corr_snps
    
    mask = snps_meta_data["names"].isin(least_corr)
    snps_meta_data.loc[mask, "categories"] = "least correlated"
    snps_meta_data.loc[mask, "raw_betas"] = effect_least_corr_snps
    
    mask = snps_meta_data["names"].isin([chosen_snp])
    snps_meta_data.loc[mask, "categories"] = "chosen snp"
    snps_meta_data.loc[mask, "raw_betas"] = effect_chosen_snp
    
    mask = snps_meta_data["names"].isin(poly_snps)
    snps_meta_data.loc[mask, "categories"] = "polygenic noise"
    snps_meta_data.loc[mask, "raw_betas"] = effect_polygenic_noise
    return snps_meta_data