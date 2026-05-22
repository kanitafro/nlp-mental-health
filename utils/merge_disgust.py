"""
Merges:
- disgust_paraphrased.csv (created by disgust_paraphrase.py)
    - method=original (original samples)
    - method=t5_paraphrase (paraphrased samples)
- disgust_backtranslated.csv (created by disgust_backtranslate.py)
    - method=backtrans_fr (back-translated samples using French)

Saves the merged dataset as disgust_all.csv, which is then used in merge_to_6labels_7labels.py 
to create the final 7-label dataset.
"""

import pandas as pd
import numpy as np

path_paraphrased = "../data/raw/disgust_paraphrased.csv"
path_backtranslated = "../data/raw/disgust_backtranslated.csv"
path_merged = "../data/raw/disgust_all.csv"

def merge_disgust():
    df1 = pd.read_csv(path_paraphrased)
    df2 = pd.read_csv(path_backtranslated)

    df1 = df1[['text', 'label', 'method']]
    df2 = df2[['text', 'label', 'method']]

    print("Value counts for disgust_paraphrased:\n", df1['method'].value_counts())

    # Taking only the French samples (works the best)
    df2 = df2[df2['method'] == 'backtrans_fr']

    print("Number of original samples:", df1[df1['method'] == 'original'].shape[0])
    print("Number of paraphrased samples:", df1[df1['method'] == 't5_paraphrase'].shape[0])
    print("Number of backtranslated samples:", df2.shape[0])

    df_merged = pd.concat([df1, df2], ignore_index=True)
    print("Total number of samples after merging:", df_merged.shape[0])
    print("Value counts for merged dataset:\n", df_merged['method'].value_counts())

    df_merged.to_csv(path_merged, index=False)
    print("Merged dataset saved to ../data/raw/disgust_all.csv")
    return df_merged

if __name__ == "__main__":
    merge_disgust()