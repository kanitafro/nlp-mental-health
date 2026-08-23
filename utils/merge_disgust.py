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
    print("-----Merging disgust datasets-----")
    #df1 = pd.read_csv(path_paraphrased)
    df2 = pd.read_csv(path_backtranslated)

    #print("Value counts for disgust_paraphrased:\n", df1['method'].value_counts())
    #print("Value counts for disgust_backtranslated:\n", df2['method'].value_counts())

    # Taking only the French samples (works the best)
    only_backtranslated = df2[df2['method'] != 'original']
    only_backtranslated = only_backtranslated[only_backtranslated['method'] != 'paraphrased']
    original = df2[df2['method'] == 'original']

    print("Number of original samples:", original.shape[0])
    print("Number of backtranslated samples:", only_backtranslated.shape[0])

    # Get 6719 instances from backtrans_de with unique original_id
    de_mixed = only_backtranslated[only_backtranslated["method"] == "backtrans_de"].drop_duplicates(subset=["original_id"]).head(6719)

    # Get backtrans_it, backtrans_es, backtrans_fr with unique original_id
    it_unique = only_backtranslated[only_backtranslated["method"] == "backtrans_it"].drop_duplicates(subset=["original_id"])
    es_unique = only_backtranslated[only_backtranslated["method"] == "backtrans_es"].drop_duplicates(subset=["original_id"])
    fr_unique = only_backtranslated[only_backtranslated["method"] == "backtrans_fr"].drop_duplicates(subset=["original_id"])

    # Calculate target count per language (33.33% of 6719)
    target_count = 6719 // 3

    # Take equal samples from each language
    it_mixed = it_unique.head(target_count)
    es_mixed = es_unique.head(target_count)
    fr_mixed = fr_unique.head(target_count)

    # Combine all
    mixed = pd.concat([original, de_mixed, it_mixed, es_mixed, fr_mixed], ignore_index=True)

    print("Checking NaN values in merged dataset:\n", mixed.isna().sum())

    # Fill NaNs
    mixed.loc[mixed['method'] == 'original', 'original_id'] = mixed.loc[mixed['method'] == 'original', 'sample_id']
    mixed.loc[mixed['method'] == 'original', 'is_augmented'] = False
    mixed.loc[mixed['method'] != 'original', 'is_augmented'] = True

    mixed["original_id"] = mixed["original_id"].astype(int)

    print("\nAfter filling NaN values:\n", mixed.isna().sum())

    print(f"\nTotal samples in mixed dataset: {len(mixed)}")
    print(f"\nMethod distribution:")
    print(mixed["method"].value_counts())
    print(f"\nUnique original_ids per method:")
    print(mixed.groupby("method")["original_id"].nunique())

    mixed.to_csv(path_merged, index=False)
    print("Merged dataset saved to ../data/raw/disgust_all.csv")
    return mixed

if __name__ == "__main__":
    merge_disgust()