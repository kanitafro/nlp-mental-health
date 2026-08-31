"""Add a reproducible train/val/test split column to the cleaned 7-label dataset.

This script mirrors the split logic in *bert/train.py*:
- non-disgust samples are split stratified by label
- disgust samples are split by original_id groups so all augmented variants stay together

The output is the same dataset with one extra column named `split` whose values are
`train`, `val`, or `test`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "processed" / "dataset_7labels_clean.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "dataset_7labels_clean_split.csv"


def assign_exact_splits(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15) -> pd.DataFrame:
    """Assign the exact train/val/test split used by bert/train.py.

    The returned dataframe keeps all original columns unchanged and adds only:
    - split: train / val / test
    """
    required_columns = {"label", "original_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.copy()

    # Match the small label normalizations in bert/train.py.
    df["label"] = df["label"].replace({"suprise": "surprise", "sad": "sadness"})

    disgust_df = df[df["label"] == "disgust"].copy()
    non_disgust_df = df[df["label"] != "disgust"].copy()

    holdout_size = val_size + test_size
    if holdout_size <= 0 or holdout_size >= 1:
        raise ValueError("val_size + test_size must be between 0 and 1.")

    # 1) Non-disgust: stratified split by label.
    train_non_disgust, holdout_non_disgust = train_test_split(
        non_disgust_df,
        test_size=holdout_size,
        stratify=non_disgust_df["label"],
        random_state=42,
    )

    test_frac_in_holdout = test_size / holdout_size
    val_non_disgust, test_non_disgust = train_test_split(
        holdout_non_disgust,
        test_size=test_frac_in_holdout,
        stratify=holdout_non_disgust["label"],
        random_state=42,
    )

    # 2) Disgust: split by original_id groups.
    if not disgust_df.empty:
        unique_original_ids = disgust_df["original_id"].unique()

        train_original_ids, holdout_original_ids = train_test_split(
            unique_original_ids,
            test_size=holdout_size,
            random_state=42,
        )

        val_original_ids, test_original_ids = train_test_split(
            holdout_original_ids,
            test_size=test_frac_in_holdout,
            random_state=42,
        )

        train_disgust = disgust_df[disgust_df["original_id"].isin(train_original_ids)].copy()
        val_disgust = disgust_df[disgust_df["original_id"].isin(val_original_ids)].copy()
        test_disgust = disgust_df[disgust_df["original_id"].isin(test_original_ids)].copy()
    else:
        train_disgust = pd.DataFrame(columns=df.columns)
        val_disgust = pd.DataFrame(columns=df.columns)
        test_disgust = pd.DataFrame(columns=df.columns)

    # 3) Merge back and annotate the split.
    train_df = pd.concat([train_non_disgust, train_disgust], ignore_index=True)
    val_df = pd.concat([val_non_disgust, val_disgust], ignore_index=True)
    test_df = pd.concat([test_non_disgust, test_disgust], ignore_index=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Restore original row order so the output is easy to compare with the source file.
    if "sample_id" in split_df.columns:
        split_df = split_df.sort_values("sample_id").reset_index(drop=True)

    return split_df

# ============================================================
# Save the split into 3 csv files
# ===========================================================
def save_split_csvs(split_df: pd.DataFrame, output_dir: Path) -> None:
    """Save the train/val/test splits into separate CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_subset = split_df[split_df["split"] == split]
        output_path = output_dir / f"{split}.csv"
        split_subset.to_csv(output_path, index=False)
        print(f"Saved {split} split to: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the exact train/val/test split used by bert/train.py to the 7-label dataset."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to dataset_7labels_clean.csv",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the split-annotated CSV.",
    )
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    df = pd.read_csv(args.input_path)
    split_df = assign_exact_splits(df, val_size=args.val_size, test_size=args.test_size)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output_path, index=False)

    print(f"Saved split-annotated dataset to: {args.output_path}")
    print(split_df["split"].value_counts())
    print("Columns in the output dataset:", list(split_df.columns))

    save_split_csvs(split_df, args.output_path.parent)
    print("Saved train/val/test CSV files in:", args.output_path.parent)

if __name__ == "__main__":
    main()