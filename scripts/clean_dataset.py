# run 'python -m scripts.clean_dataset' on terminal

import pandas as pd
from tqdm import tqdm
from preprocessing.clean_text import clean_text
from preprocessing.map_emotions import REVERSE_MAP, map_emotions_df_column

tqdm.pandas()

def clean_df(df):
    # Apply all cleaning versions
    print("Cleaning text for ML mode without NER tags:")
    df["clean_text_ml"] = df["text"].progress_apply(
        lambda x: clean_text(str(x), mode="ml")
    )

    print("Cleaning text for ML mode with NER tags:")
    df["clean_text_ml_ner"] = df["text"].progress_apply(
        lambda x: clean_text(str(x), mode="ml", use_ner_tags=True)
    )

    print("Cleaning text for transformer mode without NER tags:")
    df["clean_text_transf"] = df["text"].progress_apply(
        lambda x: clean_text(str(x), mode="transformer")
    )

    print("Cleaning text for transformer mode with NER tags:")
    df["clean_text_transf_ner"] = df["text"].progress_apply(
        lambda x: clean_text(str(x), mode="transformer", use_ner_tags=True)
    )

    print(f"Number of rows before dropping any NaNs: {df.shape[0]}")

    df.dropna(subset=['clean_text_ml'], inplace=True)
    print(f"Number of rows after dropping NaNs in 'clean_text_ml': {df.shape[0]}")

    df.dropna(subset=['clean_text_ml_ner'], inplace=True)
    print(f"Number of rows after dropping NaNs in 'clean_text_ml_ner': {df.shape[0]}")

    df.dropna(subset=['clean_text_transf'], inplace=True)
    print(f"Number of rows after dropping NaNs in 'clean_text_transf': {df.shape[0]}")

    df.dropna(subset=['clean_text_transf_ner'], inplace=True)
    print(f"Number of rows after dropping NaNs in 'clean_text_transf_ner': {df.shape[0]}")


    return df

def run_cleaning_pipeline():
    # Load datasets
    print("Loading dataset (6 emotions)...")
    df = pd.read_csv("data/raw/dataset_6labels.csv")

    print("Loading dataset (love & surprise)...")
    df_love_surprise = pd.read_csv("data/raw/love_surprise_bonus.csv")

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
    df['label'] = df['label'].replace('suprise', 'surprise')
    print("Labels: ", df['label'].unique())

    print("=== Clean 6 labels dataset ===")
    df = clean_df(df)

    print("\n=== Clean love & surprise dataset ===")
    print("Labels: ", df_love_surprise['label'].unique())
    df_love_surprise = clean_df(df_love_surprise)

    # Save output
    df.to_csv("data/processed/dataset_6labels_clean.csv", index=False)
    print("Saved cleaned dataset → data/processed/dataset_6labels_clean.csv")

    df_merged = pd.concat([df, df_love_surprise], ignore_index=True)
    df_merged.to_csv("data/processed/dataset_6labels_clean_more.csv", index=False)
    print("\nMerged dataset size: ", len(df_merged))
    print("Saved merged cleaned dataset → data/processed/dataset_6labels_clean_more.csv")

    # uncomment when you incorporate GoEmotions dataset ⬇️
    """
    print("Mapping 27 emotions from GoEmotions dataset to 6...")
    df = pd.read_csv("data/raw/goemotions.csv")

    # convert 27 → 6 classes
    df = map_emotions_df_column(df, col_name="emotion_27", new_col="emotion_6")

    df.to_csv("data/processed/goemotions_6.csv", index=False)
    """

if __name__=="__main__":
    run_cleaning_pipeline()
