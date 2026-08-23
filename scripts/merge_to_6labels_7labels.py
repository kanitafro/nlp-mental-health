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
path_raw_disgust_original = dir_raw / "disgust_original.csv"
path_augmented_disgust = dir_raw / "disgust_backtranslated.csv"

path_save_disgust = dir_raw / "disgust_all.csv"
# ------------------------------------------------------------------
# Helper function 
# ------------------------------------------------------------------
def create_ids(mode="all"):
    """Create following columns for both datasets:

    - sample_id: unique identifier for each sample (integer)
    - original_id: the original sample_id before merging (same as sample_id for original samples, can be used to track augmented samples in the future)
    - is_augmented: boolean flag indicating whether the sample is augmented (False for all current samples, but can be set to True for future augmented samples)
    - method: string indicating the method of augmentation (e.g., "original" for original samples, "backtranslation" for backtranslated samples, etc.)
    
    Args: 'mode'=
            "6labels" to create IDs for 6-label dataset, 
            "disgust" to create IDs for disgust dataset, 
            "all" to create IDs for both datasets and return both dataframes
    Returns:
        If mode=="6labels": returns df_6emos with IDs
        If mode=="disgust": returns df_disgust with IDs
        If mode=="all": returns both df_6emos and df_disgust with IDs
    """
    # Load datasets and normalize columns
    df1 = normalize_text_label_columns(pd.read_csv(path_raw_6labels))
    df2 = normalize_text_label_columns(pd.read_csv(path_raw_love_surprise))
    df_disgust = normalize_text_label_columns(pd.read_csv(path_augmented_disgust))

    df_6emos = pd.concat([df1, df2], ignore_index=True)
    df_6emos['label'] = df_6emos['label'].replace('suprise', 'surprise')
    df_6emos['label'] = df_6emos['label'].replace('sad', 'sadness')

    df_6emos = df_6emos.copy()
    df_disgust = df_disgust.copy()

    if mode == "6labels":
        # Create sample_id for 6-label dataset (e.g., 0, 1, 2, ...)
        df_6emos["sample_id"] = range(0, len(df_6emos))
        df_6emos["sample_id"] = df_6emos["sample_id"].astype(int)

        df_6emos["original_id"] = df_6emos["sample_id"]
        df_6emos["original_id"] = df_6emos["original_id"].astype(int)
        
        df_6emos["is_augmented"] = False
        df_6emos["method"] = "original"

        cols = ['sample_id', 'original_id'] + [col for col in df_6emos.columns if col not in ['sample_id', 'original_id']]
        df_6emos = df_6emos[cols]

        return df_6emos
    elif mode=='disgust':
        # Create sample_id for disgust dataset (no overlap with 6-label dataset, e.g., if 6-label dataset has 1000 samples, start from 1000)
        start_id = len(df_6emos)
        df_disgust["sample_id"] = range(start_id, start_id + len(df_disgust))
        df_disgust["original_id"] = df_disgust["sample_id"]
        df_disgust["is_augmented"] = False
        df_disgust["method"] = "original"

        cols = ['sample_id', 'original_id'] + [col for col in df_disgust.columns if col not in ['sample_id', 'original_id']]
        df_disgust = df_disgust[cols]

        return df_disgust
    elif mode == "all":
        df_6emos["sample_id"] = range(0, len(df_6emos))
        df_6emos["sample_id"] = df_6emos["sample_id"].astype(int)

        df_6emos["original_id"] = df_6emos["sample_id"]
        df_6emos["original_id"] = df_6emos["original_id"].astype(int)

        df_6emos["is_augmented"] = False
        df_6emos["method"] = "original"

        start_id = len(df_6emos)
        df_disgust["sample_id"] = range(start_id, start_id + len(df_disgust))
        df_disgust["original_id"] = df_disgust["sample_id"]
        df_disgust["is_augmented"] = False
        df_disgust["method"] = "original"

        # Fill original_id with sample_id where method is 'original'
        #df_7labels.loc[df_7labels['method'] == 'original', 'original_id'] = df_7labels.loc[df_7labels['method'] == 'original', 'sample_id']


        # Reorder columns to have sample_id and original_id at the front
        cols = ['sample_id', 'original_id'] + [col for col in df_6emos.columns if col not in ['sample_id', 'original_id']]
        df_6emos = df_6emos[cols]

        cols = ['sample_id', 'original_id'] + [col for col in df_disgust.columns if col not in ['sample_id', 'original_id']]
        df_disgust = df_disgust[cols]

        return df_6emos, df_disgust
    
def print_label_distribution(df):
    """Print the count and percentage of each label in the 'label' column of the given dataframe."""
    value_counts = df['label'].value_counts()
    percentages = df['label'].value_counts() / len(df) * 100

    result = pd.DataFrame({
        'Count': value_counts,
        'Percentage': percentages
    })

    result['Label counts'] = result['Count'].astype(str) + ' (' + result['Percentage'].round(4).astype(str) + '%)'
    print(result[['Label counts']])

# ------------------------------------------------------------------
# Merge functions
# ------------------------------------------------------------------
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

    df = create_ids(mode="6labels")
    #df = normalize_text_label_columns(pd.read_csv(path_raw_6labels))

    #print("Loading dataset (love & surprise)...")
    #df_love_surprise = normalize_text_label_columns(pd.read_csv(path_raw_love_surprise))

    """df_merged = pd.concat([df, df_love_surprise], ignore_index=True)
    df_merged['label'] = df_merged['label'].replace('suprise', 'surprise')
    df_merged['label'] = df_merged['label'].replace('sad', 'sadness')"""
    
    path_6labels_more = dir_raw / "dataset_6labels_more.csv"
    df.to_csv(path_6labels_more, index=False)

    print("NaN values in merged 6-label dataset:\n", df.isna().sum())

    print("\nMerged dataset size: ", len(df))
    print(f"Value counts in 'label' column:")
    print_label_distribution(df)
    print(f"\nColumns: {df.columns}")
    print("Sample of 6-label dataset:\n", df.head())
    print(f"Saved merged 6-emotions dataset \n     -> {path_6labels_more}")

def merge_to_7labels():
    print("\n" + "=" * 50)
    print("Creating merged 7-label dataset")
    print("=" * 50)
    print("Loading full dataset (6 labels)...")
    df6 = create_ids(mode="6labels")

    #print("Loading dataset (love & surprise)...")
    #df_love_surprise = normalize_text_label_columns(pd.read_csv(path_raw_love_surprise))

    # If path to disgust not found then run the merge_disgust() to create the merged disgust dataset

    if not path_save_disgust.exists():
        print(f"Path to disgust dataset not found at {path_save_disgust}.\nRunning merge_disgust() to create the merged disgust dataset...\n")
        df_disgust = merge_disgust()
        print("-----Finished merging disgust datasets-----")
    else:
        print("Loading dataset (disgust)...")
        df_disgust = pd.read_csv(path_save_disgust)

    df_merged = pd.concat([df6, df_disgust], ignore_index=True)
    df_merged['label'] = df_merged['label'].replace('suprise', 'surprise')
    df_merged['label'] = df_merged['label'].replace('sad', 'sadness')

    # Fill NaNs
    df_merged.loc[df_merged['method'] == 'original', 'original_id'] = df_merged.loc[df_merged['method'] == 'original', 'sample_id']
    #df_merged.loc[(df_merged['method'] == 'original') & (df_merged['label'] == 'disgust'), 'is_augmented'] = False
    df_merged.loc[df_merged['label'] != 'disgust', 'is_augmented'] = False
    df_merged.loc[df_merged['method'] == 'original', 'is_augmented'] = False
    df_merged.loc[df_merged['method'] != 'original', 'is_augmented'] = True
    print("\nNaN values in merged 7-label dataset:\n", df_merged.isna().sum())

    path_7labels = dir_raw / "dataset_7labels_ids.csv"
    df_merged.to_csv(path_7labels, index=False)

    print("\nMerged dataset size: ", len(df_merged))
    print(f"Value counts in 'label' column:")
    print_label_distribution(df_merged)
    print(f"\nColumns: {df_merged.columns}")
    print("Sample of 7-label dataset:\n", df_merged.head())
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