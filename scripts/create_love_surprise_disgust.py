# -*- coding: utf-8 -*-
"""
create_love_surprise_disgust.py

This script merges multiple datasets for love, surprise, and disgust emotions
from various sources including Kaggle datasets, Semeval, and custom datasets.
"""

import pandas as pd
import os

# ============================================
# PATH DEFINITIONS
# ============================================

# Input paths for Love & Surprise datasets
PATH_LOVE_SURPRISE_1 = "../data/raw/extra_dataset_Praveen.csv"
PATH_LOVE_SURPRISE_2 = "../data/raw/extra_dataset_PashupatiGupta.csv"
PATH_LOVE_SURPRISE_3 = "../data/raw/extra_dataset_SimaAnjali.csv"
PATH_LOVE_SURPRISE_4 = "../data/raw/extra_dataset_BryanHuerta.csv"
PATH_LOVE_SURPRISE_CUSTOM = "../data/raw/more_surprise.csv"

# Output path for merged Love & Surprise dataset
PATH_LOVE_SURPRISE_OUTPUT = "../data/raw/love_surprise_bonus.csv"

# Input paths for Disgust datasets
PATH_DISGUST_KAGGLE = "../data/raw/muhammadummarattique_data/disgust"
PATH_DISGUST_ISEAR = "hf://datasets/gsri-18/ISEAR-dataset-complete/ISEAR_dataset_complete.csv"
PATH_DISGUST_SEMEVAL_BASE_URL = "https://raw.githubusercontent.com/cbaziotis/ntua-slp-semeval2018/master/datasets/task1/E-c/"

# Output path for merged Disgust dataset
PATH_DISGUST_OUTPUT = "../data/raw/disgust_original.csv"

# Semeval file names
SEMEVAL_FILES = {
    "train": "E-c-En-train.txt",
    "dev": "E-c-En-dev.txt",
    "test": "E-c-En-test-gold.txt"
}

# Label mappings for Bryan Huerta dataset
LABELS_TO_SURPRISE = ['Surprise', 'Awe', 'Confusion', 'Arousal', 'Thrill', 
                      'Adrenaline', 'Wonder', 'Energy', 'Suspense']
LABELS_TO_LOVE = ['Gratitude', 'Love', 'Adoration', 'Admiration']

# ============================================


def merge_love_surprise_datasets():
    """Merge all love and surprise datasets"""
    
    print("=" * 50)
    print("Processing Love and Surprise Datasets")
    print("=" * 50)
    
    # Load LoSu 1 (Kaggle: Praveen Govi)
    print("\nLoading LoSu 1 (Praveen Govi dataset)...")
    df1 = pd.read_csv(PATH_LOVE_SURPRISE_1)
    df1 = df1[df1['label'].isin(['love', 'surprise'])]
    print(f"  Shape: {df1.shape}")
    print(f"  Value counts:\n{df1['label'].value_counts()}")
    
    # Load LoSu 2 (Kaggle: Pashupati Gupta)
    print("\nLoading LoSu 2 (Pashupati Gupta dataset)...")
    df2 = pd.read_csv(PATH_LOVE_SURPRISE_2)
    df2 = df2.drop('tweet_id', axis=1)
    df2 = df2.rename(columns={'sentiment': 'label', 'content': 'text'})
    df2 = df2[df2['label'].isin(['love', 'surprise', 'enthusiasm'])]
    df2['label'] = df2['label'].replace('enthusiasm', 'surprise')
    print(f"  Shape: {df2.shape}")
    print(f"  Value counts:\n{df2['label'].value_counts()}")
    
    # Load LoSu 3 (Kaggle: Sima Anjali)
    print("\nLoading LoSu 3 (Sima Anjali dataset)...")
    df3 = pd.read_csv(PATH_LOVE_SURPRISE_3)
    df3 = df3.drop('Unnamed: 0', axis=1)
    df3 = df3.rename(columns={'Emotion': 'label'})
    df3 = df3[df3['label'].isin(['love', 'surprise'])]
    print(f"  Shape: {df3.shape}")
    print(f"  Value counts:\n{df3['label'].value_counts()}")
    
    # Load LoSu 4 (Kaggle: Bryan Huerta)
    print("\nLoading LoSu 4 (Bryan Huerta dataset)...")
    df4 = pd.read_csv(PATH_LOVE_SURPRISE_4)
    df4 = df4[['Text', 'Sentiment']]
    df4 = df4.rename(columns={'Text': 'text', 'Sentiment': 'label'})
    
    df4_copy = df4[df4['label'].isin(LABELS_TO_SURPRISE + LABELS_TO_LOVE)].copy()
    df4_copy['label'] = df4_copy['label'].replace(LABELS_TO_SURPRISE, 'surprise')
    df4_copy['label'] = df4_copy['label'].replace(LABELS_TO_LOVE, 'love')
    print(f"  Shape: {df4_copy.shape}")
    print(f"  Value counts:\n{df4_copy['label'].value_counts()}")
    
    # Merge all datasets so far (df1, df2, df3, df4_copy)
    df_merged = pd.concat([df1, df2, df3, df4_copy], ignore_index=True)
    print(f"\nAfter merging all datasets: {df_merged.shape}")
    
    # Load custom surprise dataset
    print("\nLoading custom surprise dataset...")
    df_surprise = pd.read_csv(PATH_LOVE_SURPRISE_CUSTOM)
    print(f"  Shape: {df_surprise.shape}")
    
    # Merge with surprise dataset
    df_merged = pd.concat([df_merged, df_surprise], ignore_index=True)
    print(f"\nAfter merging surprise dataset: {df_merged.shape}")
    
    # Final cleanup
    df_merged.drop_duplicates(subset=['text'], inplace=True)
    print(f"Final shape after removing duplicates: {df_merged.shape}")
    print(f"Final value counts:\n{df_merged['label'].value_counts()}")
    
    # Save to CSV
    df_merged.to_csv(PATH_LOVE_SURPRISE_OUTPUT, index=False)
    print(f"\nCSV file created successfully: {PATH_LOVE_SURPRISE_OUTPUT}")
    
    return df_merged


def merge_disgust_datasets():
    """Merge all disgust datasets"""
    
    print("\n" + "=" * 50)
    print("Processing Disgust Datasets")
    print("=" * 50)
    
    # Load Disgust 1 (Kaggle: Muhammad Umar Attique)
    print("\nLoading Disgust 1 (Muhammad Umar Attique dataset)...")
    disgust_texts = []
    
    if os.path.exists(PATH_DISGUST_KAGGLE):
        for filename in os.listdir(PATH_DISGUST_KAGGLE):
            if filename.endswith('.txt'):
                filepath = os.path.join(PATH_DISGUST_KAGGLE, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    disgust_texts.append({'text': content, 'label': 'disgust'})
        
        df1_d = pd.DataFrame(disgust_texts)
        print(f"  Shape: {df1_d.shape}")
    else:
        print(f"  Warning: Path not found - {PATH_DISGUST_KAGGLE}")
        df1_d = pd.DataFrame(columns=['text', 'label'])
    
    # Load Disgust 2 (ISEAR dataset from Hugging Face)
    print("\nLoading Disgust 2 (ISEAR dataset)...")
    try:
        df2_d = pd.read_csv(PATH_DISGUST_ISEAR)
        df2_d = df2_d[df2_d['emotion'] == 'disgust']
        df2_d = df2_d.rename(columns={'emotion': 'label', 'content': 'text'})
        df2_d.drop('Unnamed: 2', axis=1, inplace=True)
        print(f"  Shape: {df2_d.shape}")
    except Exception as e:
        print(f"  Error loading ISEAR dataset: {e}")
        df2_d = pd.DataFrame(columns=['text', 'label'])
    
    # Load Disgust 3 (Semeval-2018 Task E-c)
    print("\nLoading Disgust 3 (Semeval-2018 dataset)...")
    dataframes = []
    for split, filename in SEMEVAL_FILES.items():
        try:
            url = PATH_DISGUST_SEMEVAL_BASE_URL + filename
            df = pd.read_csv(url, sep='\t', header=0, encoding='utf-8')
            df['split'] = split
            dataframes.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {filename}: {e}")
    
    if dataframes:
        df3_d = pd.concat(dataframes, ignore_index=True)
        df3_d = df3_d.rename(columns={"Tweet": "text"})
        df3_d = df3_d[df3_d["disgust"].astype(str).eq("1")].copy()
        df3_d["label"] = "disgust"
        df3_d = df3_d[["text", "label"]]
        df3_d.drop_duplicates(subset=['text'], inplace=True)
        print(f"  Shape: {df3_d.shape}")
    else:
        df3_d = pd.DataFrame(columns=['text', 'label'])
    
    # Merge all disgust datasets
    print("\nMerging all disgust datasets...")
    df_merged_d = pd.concat([df1_d, df2_d, df3_d], ignore_index=True)
    print(f"Before removing duplicates: {df_merged_d.shape[0]} rows")
    
    df_merged_d.drop_duplicates(subset=['text'], inplace=True)
    print(f"After removing duplicates: {df_merged_d.shape[0]} rows")
    
    # Save to CSV
    df_merged_d.to_csv(PATH_DISGUST_OUTPUT, index=False)
    print(f"\nSaved merged disgust dataset -> {PATH_DISGUST_OUTPUT}")
    
    return df_merged_d


def create_LoSuDi():
    """Main function to run all merging operations"""
    
    print("Starting dataset merging process...")
    print("-" * 50)
    
    # Create directories if they don't exist
    os.makedirs("../data/raw", exist_ok=True)
    
    # Merge love and surprise datasets
    love_surprise_df = merge_love_surprise_datasets()
    
    # Merge disgust datasets
    disgust_df = merge_disgust_datasets()
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Love & Surprise dataset: {love_surprise_df.shape[0]} rows")
    print(f"Disgust dataset: {disgust_df.shape[0]} rows")
    print("\nAll datasets have been processed and saved successfully!")


if __name__ == "__main__":
    create_LoSuDi()