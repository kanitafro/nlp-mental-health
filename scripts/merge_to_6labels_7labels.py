"""
This script merges the datasets to create the final 6-label and 7-label datasets.
- For the 6-label dataset, it merges:
    - dataset_6labels.csv (the original 6-label dataset)
    - love_surprise_bonus.csv (the additional love and surprise samples)
        run: python merge_to_6labels_7labels.py --label_num 6

- For the 7-label dataset, it merges:
    - dataset_6labels.csv (the original 6-label dataset)
    - love_surprise_bonus.csv (the additional love and surprise samples)
    - disgust_all.csv (the merged disgust dataset)
        run: python merge_to_6labels_7labels.py --label_num 7 

The merged datasets are saved as:
    - dataset_6labels_more.csv (the merged 6-label dataset)
    - dataset_7labels.csv (the merged 7-label dataset)
"""

import argparse
import pandas as pd
from utils.augment_disgust import augment_disgust

path_raw_6labels = "../data/raw/dataset_6labels.csv"
path_raw_love_surprise = "../data/raw/love_surprise_bonus.csv"
path_raw_disgust = "../data/raw/disgust_all.csv"

dir_raw = "../data/raw"

def merge_to_6labels():
    print("Loading full dataset (6 labels)...")
    df = pd.read_csv(path_raw_6labels)

    print("Loading dataset (love & surprise)...")
    df_love_surprise = pd.read_csv(path_raw_love_surprise)

    df_merged = pd.concat([df, df_love_surprise], ignore_index=True)
    path_6labels_more = f"{dir_raw}/dataset_6labels_more.csv"
    df_merged.to_csv(path_6labels_more, index=False)

    print("\nMerged dataset size: ", len(df_merged))
    print(f"Value counts in 'label' column: {df_merged['label'].value_counts()}")
    print(f"Saved merged 6-emotions dataset -> {path_6labels_more}")

def merge_to_7labels():
    print("Loading full dataset (6 labels)...")
    df = pd.read_csv(path_raw_6labels)

    print("Loading dataset (love & surprise)...")
    df_love_surprise = pd.read_csv(path_raw_love_surprise)

    # If path to disgust not found then run the augment_disgust() to create the merged disgust dataset
    if not path_raw_disgust:
        print(f"Path to disgust dataset not found at {path_raw_disgust}. Running augment_disgust() to create the merged disgust dataset...\n")
        augment_disgust()

    print("Loading dataset (disgust)...")
    df_disgust = pd.read_csv(path_raw_disgust)
    

    df_merged = pd.concat([df, df_love_surprise, df_disgust], ignore_index=True)
    path_7labels = f"{dir_raw}/dataset_7labels.csv"
    df_merged.to_csv(path_7labels, index=False)

    print("\nMerged dataset size: ", len(df_merged))
    print(f"Value counts in 'label' column: {df_merged['label'].value_counts()}")
    print(f"Saved merged 7-emotions dataset -> {path_7labels}")

if __name__ == "__main__":
    # Parse arguments (--label_num can be 6 or 7, if the argument is not provided, both merges will be run)
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_num', type=int, choices=[6, 7], default=None, help="Number of labels to merge (6 or 7). If not provided, both merges will be run.")
    args = parser.parse_args()

    if args.label_num == 6:
        merge_to_6labels()
    elif args.label_num == 7:
        merge_to_7labels()
    else:
        merge_to_6labels()
        merge_to_7labels()