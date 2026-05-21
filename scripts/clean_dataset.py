# run 'python -m scripts.clean_dataset' on terminal

import argparse

import pandas as pd
from tqdm import tqdm
from preprocessing.clean_text import clean_text
from preprocessing.map_emotions import REVERSE_MAP, map_emotions_df_column

tqdm.pandas()

path_raw_goemotions = "data/raw/goemotions_processed.csv"
path_raw_love_surprise = "data/raw/love_surprise_bonus.csv"
#path_raw_disgust = "data/raw/disgust_bonus.csv"
path_raw_6labels = "data/raw/dataset_6labels_more.csv"
path_raw_7labels = "data/raw/dataset_7labels.csv"

dir_cleaned = "data/processed"
path_cleaned_6labels = f"{dir_cleaned}/dataset_6labels_clean.csv"
path_cleaned_6labels_more = f"{dir_cleaned}/dataset_6labels_clean_more.csv"
path_cleaned_goemotions = f"{dir_cleaned}/goemotions.csv"
path_cleaned_7labels = f"{dir_cleaned}/dataset_7labels_clean.csv"


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

def run_cleaning_pipeline_6emotions():
    # Load datasets
    print("Loading dataset (6 emotions)...")
    df = pd.read_csv(path_raw_6labels)

    print("Loading dataset (love & surprise)...")
    df_love_surprise = pd.read_csv(path_raw_love_surprise)

    print("Loading dataset (disgust)...")
    df_disgust = pd.read_csv(path_raw_disgust)

    print(f"Loaded dataset_6labels.csv with {len(df)} rows")

    # rename columns of full dataset to be 'text' and 'label'
    df = df.rename(columns={'sentence': 'text', 'emotion': 'label'})
    print("Column names after renaming:", end=" ")
    for i, col in enumerate(df.columns):
        if i < len(df.columns)-1:
            print(col, end=", ")
        else:
            print(col)
    
    # rename label 'sad' to 'sadness'
    df['label'] = df['label'].replace('sad', 'sadness')    
    # rename label 'suprise' to 'surprise' (fix typo)
    df['label'] = df['label'].replace('suprise', 'surprise')
    print("Labels: ", df['label'].unique())

    print("=== Clean 6 labels dataset ===")
    df = clean_df(df)

    print("\n=== Clean love & surprise dataset ===")
    print("Labels: ", df_love_surprise['label'].unique())
    df_love_surprise = clean_df(df_love_surprise)

    print("\n=== Clean disgust dataset ===")
    print("Labels: ", df_disgust['label'].unique())
    df_disgust = clean_df(df_disgust)

    # Save output
    df.to_csv(path_cleaned_6labels, index=False)
    print(f"Saved cleaned 6-emotions dataset -> {path_cleaned_6labels}")

    df_merged = pd.concat([df, df_love_surprise], ignore_index=True)
    df_merged.to_csv(path_cleaned_6labels_more, index=False)
    print("\nMerged dataset size: ", len(df_merged))
    print(f"Saved merged cleaned 6-emotions dataset -> {path_cleaned_6labels_more}")

    df_merged = pd.concat([df, df_love_surprise, df_disgust], ignore_index=True)
    df_merged.to_csv(path_cleaned_7labels, index=False)
    print("\nMerged dataset size: ", len(df_merged))
    print(f"Saved merged cleaned 7-emotions dataset -> {path_cleaned_7labels}")


def run_cleaning_pipeline_goemotions():
    print("Loading dataset (GoEmotions)...")
    df_goemotions = pd.read_csv(path_raw_goemotions)

    if "text" not in df_goemotions.columns:
        raise ValueError("GoEmotions dataset must contain a 'text' column.")

    print(f"Loaded {path_raw_goemotions} with {len(df_goemotions)} rows")
    print("=== Clean GoEmotions dataset ===")
    df_goemotions = clean_df(df_goemotions)
    df_goemotions.to_csv(path_cleaned_goemotions, index=False)
    print(f"Saved cleaned GoEmotions dataset -> {path_cleaned_goemotions}")


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
        run_cleaning_pipeline_6emotions()
    elif action == "clean_goemotions":
        run_cleaning_pipeline_goemotions()
    elif action == "clean_all":
        run_cleaning_pipeline_6emotions()
        run_cleaning_pipeline_goemotions()
    else:
        raise ValueError(f"Unsupported action: {action}")


if __name__=="__main__":
    args = parse_args()
    run_cleaning_pipeline(args.action)
