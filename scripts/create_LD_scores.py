# Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from abyss.glob import glob
import pandas as pd
import numpy as np
import sys
# Import relevant functions from abyss.glob
from abyss.glob import glob

# Parse command line arguments
args = sys.argv
arguments = {
    args[1].split(':')[0]: args[1].split(':')[1]
}

chrom = int(arguments['chrom'])

path_input = glob.PATH_usefull + f"/LD_scores/chrom_{chrom}"
list_of_windows = os.listdir(path_input)
# Sorting the list based on the numeric value after "window_"
sorted_list_of_windows = sorted(list_of_windows, key=lambda x: int(x.split('_')[1].split('.')[0]))

path_window = path_input + f"/{sorted_list_of_windows[0]}"
total = pd.read_pickle(path_window)
total['counter'] = 1
for window in sorted_list_of_windows:
    path_window = path_input + f"/{window}"
    temp = pd.read_pickle(path_window)
    temp['counter'] = 1
    total = pd.merge(total, temp, on='ID', how='outer', suffixes=('_from_total', '_from_temp'))
    total['counter'] = (total['counter_from_total'] + total['counter_from_temp'])
    total['multi_old'] = total['counter_from_total']/total['counter']
    total['multi_new'] = 1/total['counter']

    begin = total[pd.isna(total['LD_score_from_temp'])].copy()
    middle = total[~pd.isna(total['LD_score_from_total']) & ~pd.isna(total['LD_score_from_temp'])].copy()
    end = total[pd.isna(total['LD_score_from_total'])].copy()

    begin['LD_score'] = begin['LD_score_from_total']
    begin['counter'] = begin['counter_from_total']
    begin = begin[['ID','LD_score','counter']]
    end['LD_score'] = end['LD_score_from_temp']
    end['counter'] = end['counter_from_temp']
    end = end[['ID','LD_score','counter']]

    middle['LD_score'] = middle['multi_old']*middle['LD_score_from_total'] + middle['multi_new']*middle['LD_score_from_temp']

    middle = middle[['ID','LD_score','counter']]
    total = pd.concat([begin, middle, end], ignore_index=True)

# Splitting the 'ID' column by ':' and expanding into separate columns
split_columns = total['ID'].str.split(':', expand=True)

# The middle number is in the second column (index 1) after splitting
total['position'] = split_columns[1]

# Optionally, convert 'position' to an integer if it's always a number
total['position'] = total['position'].astype(int)
os.system(f"rm {path_input}/*")
max_LD = np.round(total['LD_score'].max(),3)
total.to_pickle(f"{path_input}/LD_scores_max_{max_LD}.pkl")
