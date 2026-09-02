# project-root/scripts/handle_goemotions.py
"""
Download and preprocess the GoEmotions dataset.

- Raw TSV files and emotions.txt are saved to: project-root/data/raw/goemotions/
- Preprocessed CSV files (multi-hot encoded + cleaned text) are saved to:
  project-root/data/processed/goemotions/
"""

import os
import requests
import pandas as pd
from pathlib import Path

# Import your cleaning function
from scripts.clean_dataset import clean_df

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------

BASE_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"

FILES = {
    "train.tsv": "train.tsv",
    "dev.tsv": "dev.tsv",
    "test.tsv": "test.tsv",
    "emotions.txt": "emotions.txt",
}

# Project root (assumes script is in e.g. project-root/scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "goemotions"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "goemotions"

# Cleaning parameters (adjust as you like)
CLEAN_MODE = "ml"          # "ml", "transformer", or "all"
USE_NER_TAGS = False       # whether to also create clean_text_ml_ner

# ----------------------------------------------------------------------
# 2. Download raw files (no preprocessing)
# ----------------------------------------------------------------------

def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading {url} -> {dest}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"    Done ({dest.stat().st_size} bytes)")

def download_all() -> None:
    for filename, url_suffix in FILES.items():
        url = BASE_URL + url_suffix
        dest = RAW_DATA_DIR / filename
        download_file(url, dest)

# ----------------------------------------------------------------------
# 3. Preprocess data (read from raw, clean, write to processed)
# ----------------------------------------------------------------------

def load_emotions(emotions_path: Path) -> list:
    with open(emotions_path, "r") as f:
        emotions = [line.strip() for line in f if line.strip()]
    return emotions

def preprocess_tsv_to_multi_hot(tsv_path: Path, emotions: list) -> pd.DataFrame:
    """
    Read a GoEmotions TSV file and convert labels to multi‑hot binary vectors.
    Returns a DataFrame with columns: text, id, and one column per emotion.
    """
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["text", "labels", "id"])

    # Convert label string "0,1,3" -> list of ints
    df["label_indices"] = df["labels"].str.split(",").apply(lambda x: [int(i) for i in x if i])

    # Create one binary column per emotion
    for idx, emotion in enumerate(emotions):
        df[emotion] = df["label_indices"].apply(lambda indices: 1 if idx in indices else 0)

    # Drop the intermediate column and the original 'labels' string
    df.drop(columns=["label_indices", "labels"], inplace=True)

    return df

def preprocess_all() -> None:
    """Load raw TSVs, convert labels, clean text, and save processed CSVs."""
    emotions_path = RAW_DATA_DIR / "emotions.txt"
    if not emotions_path.exists():
        raise FileNotFoundError(f"    emotions.txt not found at {emotions_path}. Run download_all() first.")

    emotions = load_emotions(emotions_path)
    print(f"    Loaded {len(emotions)} emotion labels: {emotions}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        tsv_path = RAW_DATA_DIR / f"{split}.tsv"
        if not tsv_path.exists():
            print(f"    Warning: {tsv_path} not found, skipping {split}.")
            continue

        print(f"\n---- Processing {split}... ----")

        # 1) Convert labels to multi‑hot (keeps original 'text' column)
        df = preprocess_tsv_to_multi_hot(tsv_path, emotions)
        print(f"    After label conversion: {len(df)} rows")

        # 2) Apply cleaning (adds clean_text_ml and optionally clean_text_ml_ner)
        #    The function will drop rows where the cleaned text becomes NaN.
        print("    Applying cleaning...")
        df = clean_df(df, transf_ner_tags=True, ml_ner_tags=True)
        # Now df has extra columns: clean_text_ml (and possibly clean_text_ml_ner)

        # 3) (Optional) Keep only the columns you need.
        #    Here we keep: id, original text, cleaned text(s), and all emotion columns.
        #    If you want to drop the original 'text', just comment it out.
        keep_cols = ["id", "text", "clean_text_ml", "clean_text_ml_ner", "clean_text_transf", "clean_text_transf_ner"] + [emotion for emotion in emotions]
        """if CLEAN_MODE in ["ml", "all"]:
            keep_cols.append("clean_text_ml")
        if USE_NER_TAGS and CLEAN_MODE in ["ml", "all"]:
            keep_cols.append("clean_text_ml_ner")"""
        # For 'transformer' mode, you'd add 'clean_text_transformer' etc.
        # Adjust according to your clean_df implementation.

        # Ensure only existing columns are kept
        keep_cols = [col for col in keep_cols if col in df.columns]
        df = df[keep_cols]
        print("    Columns kept for output:", keep_cols)
        print(f"    After cleaning: {len(df)} rows")
        print(f"    Sample rows:\n{df.head(3)}")
        print("    Value counts for each emotion column:")
        for emotion in emotions:
            print(f"      - {emotion}: {df[emotion].sum()}")

        # 4) Save to processed directory
        csv_path = PROCESSED_DATA_DIR / f"{split}_preprocessed.csv"
        df.to_csv(csv_path, index=False)
        print(f"    Saved {len(df)} rows to {csv_path}")

def merge_all_processed() -> None:
    """Merge train/dev/test processed CSVs into a single CSV."""
    all_dfs = []
    for split in ["train", "dev", "test"]:
        csv_path = PROCESSED_DATA_DIR / f"{split}_preprocessed.csv"
        if not csv_path.exists():
            print(f"    Warning: {csv_path} not found, skipping {split}.")
            continue
        df = pd.read_csv(csv_path)
        df["split"] = split  # Add a column to indicate the split
        all_dfs.append(df)

    if not all_dfs:
        print("    No processed CSVs found to merge.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    merged_csv_path = PROCESSED_DATA_DIR / "goemotions_merged_preprocessed.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"    Merged {len(merged_df)} rows from train/dev/test into {merged_csv_path}")

# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GoEmotions Dataset Download & Preprocess (with cleaning)")
    print("=" * 60)

    print("\n[1] Downloading raw files (no preprocessing)...")
    download_all()

    print("\n[2] Preprocessing and cleaning data...")
    preprocess_all()

    print("\n[3] Merging processed data...")
    merge_all_processed()

    print("\n✅ All done!")
    print(f"Raw files (untouched):     {RAW_DATA_DIR}")
    print(f"Processed CSVs (cleaned):  {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
