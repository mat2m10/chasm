# Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from abyss.glob import glob
import pandas as pd
import numpy as np
import sys
# Import relevant functions from abyss.glob
from abyss.glob import glob

# Define the name for output files
name = "total_pop"

# Parse command line arguments
args = sys.argv
arguments = {
    args[1].split(':')[0]: args[1].split(':')[1],
    args[2].split(':')[0]: args[2].split(':')[1],
    args[3].split(':')[0]: args[3].split(':')[1]
}

window_size = int(arguments['window_size'])
chrom = int(arguments['chrom'])
step_size = int(arguments['step_size'])

# Define paths
PATH_input = glob.PATH_GENO+ "/02_per_chrom/"
PATH_output = glob.PATH_usefull + "/LD_scores/"
path_output = PATH_output+ f"/chrom_{chrom}"
os.makedirs(path_output, exist_ok=True)

print(f"Finding allele frequencies of chromosome {chrom}")
output_file = "output.txt"
os.system(f"./plink2 --bfile {PATH_input}{name}_chr_{chrom} --make-bed --out ./temp_{chrom}"+ " > " + output_file + " 2>&1")
os.system(f"./plink2 --bfile ./temp_{chrom} --freq --out ./AF_{chrom}"+ " > " + output_file + " 2>&1")
AFs_meta = pd.read_csv(f"./AF_{chrom}.afreq", sep="\t") # convert to pandas dataframe
os.system(f"rm AF_{chrom}*")
os.system(f"rm temp_{chrom}*")
os.system(f"rm output.txt")

# List to store the windows
windows = []

# Loop through the dataframe in steps of 'step_size'
for start in range(0, len(AFs_meta) - window_size + 1, step_size):
    end = start + window_size
    window = AFs_meta.iloc[start:end]
    windows.append(window)
windows.append(AFs_meta.tail(window_size))
is_in_outputs = [int(f.split("_")[1].split('.')[0]) for f in os.listdir(path_output)]
i_to_iterate = [f for f in list(range(len(windows))) if f not in is_in_outputs]
for i in i_to_iterate:
    percent = np.round((i+1)/(len(windows)+1)*100,1)
    window = windows[i]
    snp_ids_to_keep = " ,".join(list(window['ID']))
    os.system(f"./plink2 --bfile {PATH_input}{name}_chr_{chrom} --snps {snp_ids_to_keep} --mac 50 --geno 0.05  --recode A --out ./temp_{chrom}"+ " > " + output_file + " 2>&1")
    to_drop = ['FID','IID','PAT','MAT','SEX', 'PHENOTYPE']
    genos = pd.read_csv(f"./temp_{chrom}.raw", sep="\t")
    genos = genos.fillna(2)
    genos = genos.drop(to_drop, axis=1).astype(int) - 1 # Encoding it [-1, 0, 1]
    genos = genos.applymap(lambda x :x if x in [-1, 0, 1] else 1)
    genos = genos.T.drop_duplicates(keep='first').T
    r_squared_matrix = genos.corr() ** 2
    # Sum r^2 values for each SNP to get LD scores
    ld_scores = r_squared_matrix.sum()
    ld_score_df = pd.DataFrame(list(ld_scores.items()), columns=['ID', 'LD_score'])
    ld_score_df['ID'] = ld_score_df['ID'].str[:-2]
    merged_data = pd.merge(ld_score_df, window, on='ID')
    merged_data = merged_data[['ID','LD_score']]
    merged_data.to_pickle(path_output + f"/window_{i}.pkl")
    os.system(f"rm temp_{chrom}.log")
    os.system(f"rm temp_{chrom}.raw")
    print(f"done {percent}% for chrom {chrom}")
