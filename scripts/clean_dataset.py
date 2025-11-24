import pandas as pd
from tqdm import tqdm
from preprocessing.clean_text import clean_text
from preprocessing.map_emotions import REVERSE_MAP, map_emotions_df_column

tqdm.pandas()

def run_cleaning_pipeline():
    print("Loading dataset (with 6 emotions)...")

    df = pd.read_csv("data/raw/dataset_6labels.csv")

    # rename columns to be 'text' and 'label'
    df = df.rename(columns={'sentence': 'text', 'emotion': 'label'})

    # Apply all cleaning versions
    df["clean_text_ml"] = df["text"].progress_apply(
        lambda x: clean_text(str(x), mode="ml")
    )

    df["clean_text_ml_ner"] = df["text"].progress_apply(
        lambda x: clean_text(str(x), mode="ml", use_ner_tags=True)
    )

    df["clean_text_transf"] = df["text"].progress_apply(
        lambda x: clean_text(str(x), mode="transformer")
    )

    df["clean_text_transf_ner"] = df["text"].progress_apply(
        lambda x: clean_text(str(x), mode="transformer", use_ner_tags=True)
    )

    # Save output
    df.to_csv("data/processed/dataset_6labels_clean.csv", index=False)
    print("Saved cleaned dataset → data/processed/dataset_6labels_clean.csv")
    
    # uncomment when you incorporate GoEmotions dataset ⬇️
    """
    print("Mapping 27 emotions from GoEmotions dataset to 6...")
    df = pd.read_csv("data/raw/goemotions.csv")

    # convert 27 → 6 classes
    df = map_emotions_df_column(df, col_name="emotion_27", new_col="emotion_6")

    df.to_csv("data/processed/goemotions_6.csv", index=False)
    """
