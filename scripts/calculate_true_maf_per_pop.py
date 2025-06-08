import numpy as np
import pandas as pd
import subprocess
import os

def calculate_true_maf_per_pop(genos, humans):
    geno = genos.copy()
    geno['pop'] = humans['populations']

    p2s_dfs = []
    q2s_dfs = []
    twopqs_dfs = []
    
    for pop in geno['pop'].unique():
        temp = geno[geno['pop'] == pop].drop("pop", axis=1)
    
        # Count the number of major, heterozygous, and minor alleles
        counts = temp.apply(pd.Series.value_counts).fillna(0)
        try:
            num_maj = counts.loc[1.0]
        except:
            num_maj = 0
        try:
            num_het = counts.loc[0.0]
        except:
            num_het = 0
        try:
            num_min = counts.loc[-1.0]
        except:
            num_min = 0
    
        total_humans = num_maj + num_het + num_min
    
        # Normalize to get frequencies instead of counts
        p2s = num_maj / total_humans
        twopqs = num_het / total_humans
        q2s = num_min / total_humans
    
        # Expand the normalized values across all rows for each population
        p2s_dfs.append(pd.DataFrame([p2s] * temp.shape[0], index=temp.index, columns=temp.columns))
        twopqs_dfs.append(pd.DataFrame([twopqs] * temp.shape[0], index=temp.index, columns=temp.columns))
        q2s_dfs.append(pd.DataFrame([q2s] * temp.shape[0], index=temp.index, columns=temp.columns))
        
    # Drop "pop" from the original DataFrame
    geno = geno.drop("pop", axis=1)
    
    # Concatenate all population-specific DataFrames
    true_p2s = pd.concat(p2s_dfs)
    true_twopqs = pd.concat(twopqs_dfs)
    true_q2s = pd.concat(q2s_dfs)

    return true_p2s, true_twopqs, true_q2s
