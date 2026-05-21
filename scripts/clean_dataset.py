# run 'python -m scripts.clean_dataset' on terminal

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

import pandas as pd
from tqdm import tqdm
from preprocessing.clean_text import clean_text

tqdm.pandas()

path_raw_goemotions = REPO_ROOT / "data" / "raw" / "goemotions_processed.csv"
#path_raw_love_surprise = "data/raw/love_surprise_bonus.csv"
#path_raw_disgust = "data/raw/disgust_bonus.csv"
path_raw_6labels = REPO_ROOT / "data" / "raw" / "dataset_6labels_more.csv"
path_raw_7labels = REPO_ROOT / "data" / "raw" / "dataset_7labels.csv"

dir_cleaned = REPO_ROOT / "data" / "processed"
path_cleaned_6labels = dir_cleaned / "dataset_6labels_clean.csv"
path_cleaned_6labels_more = dir_cleaned / "dataset_6labels_clean_more.csv"
path_cleaned_goemotions = dir_cleaned / "goemotions.csv"
path_cleaned_7labels = dir_cleaned / "dataset_7labels_clean.csv"


def clean_df(df, mode="all", transf_ner_tags=False, ml_ner_tags=False):
    # Apply all cleaning versions based on the specified mode 
    if mode not in ["transformer", "ml", "all"]:
        raise ValueError("Invalid mode. Choose 'transformer', 'ml', or 'all'.")
    
    print(f"Number of rows before dropping any NaNs: {df.shape[0]}")

    if mode == "ml" or mode == "all":
        print("Cleaning text for ML mode without NER tags:")
        df["clean_text_ml"] = df["text"].progress_apply(
            lambda x: clean_text(str(x), mode="ml")
        )

        df.dropna(subset=['clean_text_ml'], inplace=True)
        print(f"Number of rows after dropping NaNs in 'clean_text_ml': {df.shape[0]}")

        if ml_ner_tags:
            print("Cleaning text for ML mode with NER tags:")
            df["clean_text_ml_ner"] = df["text"].progress_apply(
                lambda x: clean_text(str(x), mode="ml", use_ner_tags=True)
            )

            df.dropna(subset=['clean_text_ml_ner'], inplace=True)
            print(f"Number of rows after dropping NaNs in 'clean_text_ml_ner': {df.shape[0]}")



    if mode == "transformer" or mode == "all":
        print("Cleaning text for transformer mode without NER tags:")
        df["clean_text_transf"] = df["text"].progress_apply(
            lambda x: clean_text(str(x), mode="transformer")
        )
        df.dropna(subset=['clean_text_transf'], inplace=True)
        print(f"Number of rows after dropping NaNs in 'clean_text_transf': {df.shape[0]}")
        
        if transf_ner_tags:
            print("Cleaning text for transformer mode with NER tags:")
            df["clean_text_transf_ner"] = df["text"].progress_apply(
                lambda x: clean_text(str(x), mode="transformer", use_ner_tags=True)
            )

            df.dropna(subset=['clean_text_transf_ner'], inplace=True)
            print(f"Number of rows after dropping NaNs in 'clean_text_transf_ner': {df.shape[0]}")


    return df

def run_cleaning_pipeline_450k(input_path, output_path, num_labels=6):
    # Load datasets
    print(f"=== Clean {num_labels} labels dataset ===")
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"Loaded dataset with {len(df)} rows")

    print(f"\nLabels ({num_labels} emotions): ", df['label'].unique())
    print("Cleaning...")
    df = clean_df(df, transf_ner_tags=True, ml_ner_tags=True)
    print(f"\nCleaned dataset size: {len(df)}")

    # rename columns of full dataset to be 'text' and 'label'
    df = df.rename(columns={'sentence': 'text', 'emotion': 'label'})
    print("Column names after renaming:", end=" ")
    for i, col in enumerate(df.columns):
        if i < len(df.columns)-1:
            print(col, end=", ")
        else:
            print(col)

    df['label'] = df['label'].replace('sad', 'sadness')
    df['label'] = df['label'].replace('suprise', 'surprise')

    # Save output (6 labels)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned {num_labels}-emotions dataset -> {output_path}")


def run_cleaning_pipeline_goemotions():
    print("Loading dataset (GoEmotions)...")
    df_goemotions = pd.read_csv(path_raw_goemotions)

    if "text" not in df_goemotions.columns:
        raise ValueError("GoEmotions dataset must contain a 'text' column.")

    print(f"Loaded {path_raw_goemotions} with {len(df_goemotions)} rows")
    print("=== Clean GoEmotions dataset ===")
    df_goemotions = clean_df(df_goemotions, transf_ner_tags=True, ml_ner_tags=True)
    df_goemotions.to_csv(path_cleaned_goemotions, index=False)
    print(f"Saved cleaned GoEmotions dataset -> {path_cleaned_goemotions}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean datasets for emotion modeling.")
    parser.add_argument(
        "action",
        choices=["clean_6emotions", "clean_goemotions", "clean_all"],
        help="Choose which cleaning pipeline to run.", 
        default="clean_all"
    )
    return parser.parse_args()


def run_cleaning_pipeline(action):
    if action == "clean_6emotions":
        run_cleaning_pipeline_450k(path_raw_6labels, path_cleaned_6labels, num_labels=6)
    elif action == "clean_7emotions":
        run_cleaning_pipeline_450k(path_raw_7labels, path_cleaned_7labels, num_labels=7)
    elif action== "clean_450k":
        run_cleaning_pipeline_450k(path_raw_6labels, path_cleaned_6labels, num_labels=6)
        run_cleaning_pipeline_450k(path_raw_7labels, path_cleaned_7labels, num_labels=7)
    elif action == "clean_goemotions":
        run_cleaning_pipeline_goemotions()
    elif action == "clean_all":
        run_cleaning_pipeline_450k(path_raw_6labels, path_cleaned_6labels, num_labels=6)
        run_cleaning_pipeline_450k(path_raw_7labels, path_cleaned_7labels, num_labels=7)
        run_cleaning_pipeline_goemotions()
    else:
        raise ValueError(f"Unsupported action: {action}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Clean datasets for emotion modeling.")
    parser.add_argument('--action', type=str, choices=['clean_6emotions', 'clean_7emotions', 'clean_450k', 'clean_goemotions', 'clean_all'], default='clean_all', help="Choose which cleaning pipeline to run.")
    args = parser.parse_args()

    run_cleaning_pipeline(args.action)
