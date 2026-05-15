import pandas as pd

path_raw_6labels = "data/raw/dataset_6labels.csv"
path_raw_love_surprise = "data/raw/love_surprise_bonus.csv"
path_raw_disgust = "data/raw/disgust_bonus.csv"

dir_raw = "data/raw"

def merge_to_6labels():
    print("Loading full dataset (6 labels)...")
    df = pd.read_csv(path_raw_6labels)

    print("Loading dataset (love & surprise)...")
    df_love_surprise = pd.read_csv(path_raw_love_surprise)

    df_merged = pd.concat([df, df_love_surprise], ignore_index=True)
    path_6labels_more = f"{dir_raw}/dataset_6labels_more.csv"
    df_merged.to_csv(path_6labels_more, index=False)

    print("\nMerged dataset size: ", len(df_merged))
    print(f"Value counts in 'label' column: {df_merged['label'].value_counts()}")
    print(f"Saved merged 6-emotions dataset -> {path_6labels_more}")

def merge_to_7labels():
    print("Loading full dataset (6 labels)...")
    df = pd.read_csv(path_raw_6labels)

    print("Loading dataset (love & surprise)...")
    df_love_surprise = pd.read_csv(path_raw_love_surprise)

    print("Loading dataset (disgust)...")
    df_disgust = pd.read_csv(path_raw_disgust)

    df_merged = pd.concat([df, df_love_surprise, df_disgust], ignore_index=True)
    path_7labels = f"{dir_raw}/dataset_7labels.csv"
    df_merged.to_csv(path_7labels, index=False)

    print("\nMerged dataset size: ", len(df_merged))
    print(f"Value counts in 'label' column: {df_merged['label'].value_counts()}")
    print(f"Saved merged 7-emotions dataset -> {path_7labels}")

if __name__ == "__main__":
    merge_to_6labels()
    merge_to_7labels()