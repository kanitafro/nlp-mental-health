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
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.merge_disgust import merge_disgust

dir_raw = REPO_ROOT / "data" / "raw"
path_raw_6labels = dir_raw / "dataset_6labels.csv"
path_raw_love_surprise = dir_raw / "love_surprise_bonus.csv"
path_raw_disgust = dir_raw / "disgust_all.csv"



def normalize_text_label_columns(df):
    """Return a dataframe with a consistent text/label schema."""
    rename_map = {}
    if "sentence" in df.columns:
        rename_map["sentence"] = "text"
    if "emotion" in df.columns:
        rename_map["emotion"] = "label"

    df = df.rename(columns=rename_map)

    required_columns = ["text", "label"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df[required_columns]

def merge_to_6labels():
    print("=" * 50)
    print("Creating merged 6-label dataset")
    print("=" * 50)
    print("Loading full dataset (6 labels)...")
    df = normalize_text_label_columns(pd.read_csv(path_raw_6labels))

    print("Loading dataset (love & surprise)...")
    df_love_surprise = normalize_text_label_columns(pd.read_csv(path_raw_love_surprise))

    df_merged = pd.concat([df, df_love_surprise], ignore_index=True)
    df_merged['label'] = df_merged['label'].replace('suprise', 'surprise')
    df_merged['label'] = df_merged['label'].replace('sad', 'sadness')
    
    path_6labels_more = dir_raw / "dataset_6labels_more.csv"
    df_merged.to_csv(path_6labels_more, index=False)

    print("\nMerged dataset size: ", len(df_merged))
    print(f"Value counts in 'label' column: \n{df_merged['label'].value_counts()}")
    print("Columns: ", df_merged.columns)
    print(f"Saved merged 6-emotions dataset \n     -> {path_6labels_more}")

def merge_to_7labels():
    print("\n" + "=" * 50)
    print("Creating merged 7-label dataset")
    print("=" * 50)
    print("Loading full dataset (6 labels)...")
    df = normalize_text_label_columns(pd.read_csv(path_raw_6labels))

    print("Loading dataset (love & surprise)...")
    df_love_surprise = normalize_text_label_columns(pd.read_csv(path_raw_love_surprise))

    # If path to disgust not found then run the merge_disgust() to create the merged disgust dataset

    if not path_raw_disgust.exists():
        print(f"Path to disgust dataset not found at {path_raw_disgust}.\nRunning merge_disgust() to create the merged disgust dataset...\n")
        df_disgust = merge_disgust()
    else:
        print("Loading dataset (disgust)...")
        df_disgust = normalize_text_label_columns(pd.read_csv(path_raw_disgust))

    df_merged = pd.concat([df, df_love_surprise, df_disgust], ignore_index=True)
    df_merged['label'] = df_merged['label'].replace('suprise', 'surprise')
    df_merged['label'] = df_merged['label'].replace('sad', 'sadness')

    path_7labels = dir_raw / "dataset_7labels.csv"
    df_merged.to_csv(path_7labels, index=False)

    print("\nMerged dataset size: ", len(df_merged))
    print(f"Value counts in 'label' column: \n{df_merged['label'].value_counts()}")
    print(f"Columns: {df_merged.columns}")
    print(f"Saved merged 7-emotions dataset \n     -> {path_7labels}")

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