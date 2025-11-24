import pandas as pd

from scripts.clean_dataset import run_cleaning_pipeline
from data.lexicon.build_lexicon import build_and_save_lexicon
from preprocessing.tokenizer import fit_tfidf_vectorizer


def main():
    build_and_save_lexicon()
    print("=== Clean 6 labels dataset ===")
    run_cleaning_pipeline()

    df = pd.read_csv("data/processed/dataset_6labels_clean.csv")
    texts = df["clean_text_ml"].tolist()


    vectorizer, X = fit_tfidf_vectorizer(
        texts,
        mode="ml",
        use_ner_tags=True,
        max_features=50000,
        min_df=3,
        max_df=0.9,
        save_path="models/tfidf.joblib"
    )

    # ready for ML models
    print(X.shape)

if __name__ == "__main__":
    main()
