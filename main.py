import pandas as pd
import numpy as np

from preprocessing.tokenizer import fit_tfidf_vectorizer
from models.train_logreg_v1 import train_logreg_v1

def main():
    df = pd.read_csv("data/processed/dataset_6labels_clean.csv")
    """
    # This will be run in the models/ files
    texts = df["clean_text_ml"].tolist()

    texts = ["" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text) for text in texts]
    vectorizer, X = fit_tfidf_vectorizer(
        texts,
        mode="ml",
        use_ner_tags=False,
        max_features=50000,
        min_df=2,
        max_df=0.9,
        save_path="models/tfidf.joblib"
    )

    # ready for ML models
    print(X.shape)
    """
    # train_logreg_v1()

if __name__ == "__main__":
    main()
