# run 'python -m scripts.clean_risklabels' on terminal

'''
Cleans risk label datasets and saves cleaned versions.
Assumes input CSVs are in data/risk_labels/ and saves to data/processed/
'''
import pandas as pd
from tqdm import tqdm
from preprocessing.clean_text import clean_text

tqdm.pandas()

def clean_df(df):
    print("Labels: ", df['label'].unique())
    print("Cleaning text for transformer mode without NER tags:")
    df["clean_text_transf"] = df["text"].progress_apply(
        lambda x: clean_text(str(x), mode="transformer")
    )

    return df

def run_cleaning_pipeline():
    print("\n=== Loading dataset (risk label: depression) ===")

    df1 = pd.read_csv("data/risk_labels/reddit_depression.csv")
    print(f"Loaded reddit_depression.csv with {len(df1)} rows")
    print("Cleaning dataset...")
    df1 = clean_df(df1)

    #####################################################################
    print("\n=== Loading dataset (risk label: grief) ===")

    df2 = pd.read_csv("data/risk_labels/reddit_grief.csv")
    print(f"Loaded reddit_grief.csv with {len(df2)} rows")
    print("Cleaning dataset...")
    df2 = clean_df(df2)

    ######################################################################
    print("\n=== Loading dataset (risk label: suicidal) ===")

    df3 = pd.read_csv("data/risk_labels/reddit_suicidal.csv")
    print(f"Loaded reddit_suicidal.csv with {len(df3)} rows")
    print("Cleaning dataset...")
    df3 = clean_df(df3)
    ######################################################################
    print("\n=== Loading dataset (risk label: selfharm) ===")

    df4 = pd.read_csv("data/risk_labels/reddit_selfharm.csv")
    print(f"Loaded reddit_selfharm.csv with {len(df4)} rows")
    print("Cleaning dataset...")
    df4 = clean_df(df4)

    # Save output
    print("\n=== Saving cleaned datasets ===")

    df1.to_csv("data/processed/dataset_depression_clean.csv", index=False)
    print("Saved cleaned dataset → data/processed/dataset_depression_clean.csv")

    df2.to_csv("data/processed/dataset_grief_clean.csv", index=False)
    print("Saved cleaned dataset → data/processed/dataset_grief_clean.csv")

    df3.to_csv("data/processed/dataset_suicidal_clean.csv", index=False)
    print("Saved cleaned dataset → data/processed/dataset_suicidal_clean.csv")

    df4.to_csv("data/processed/dataset_selfharm_clean.csv", index=False)
    print("Saved cleaned dataset → data/processed/dataset_selfharm_clean.csv")

if __name__=="__main__":
    run_cleaning_pipeline()
