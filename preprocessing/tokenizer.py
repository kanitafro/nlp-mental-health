import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------------------------------------
# 1. Custom tokenizer WITH mode + NER toggle
# -----------------------------------------------------------
def custom_tokenizer(text, mode="ml", use_ner_tags=False):
    """
    Tokenizer used for ML-based vectorizers (TF-IDF).
    Transformers use their own tokenizer, so mode='transformer' bypasses this.
    """

    if mode == "transformer":
        # Transformers do NOT use custom tokenizer
        # We return raw words so TF-IDF is never used in transformer mode
        return text.split()

    tokens = []

    # Always preserve NOT_ tokens
    tokens += re.findall(r"NOT_\w+", text)

    # Optionally keep NER tags
    if use_ner_tags:
        tokens += re.findall(r"\[[A-Z_]+\]", text)

    # Regular words (underscores allowed)
    tokens += re.findall(r"[a-zA-Z_]+", text)

    return tokens


# -----------------------------------------------------------
# 2. Extract all NOT_ and NER tokens → force-keep in vocabulary
# -----------------------------------------------------------
def extract_special_tokens(texts, use_ner_tags=False):
    specials = set()

    for t in texts:
        specials.update(re.findall(r"NOT_\w+", t))
        if use_ner_tags:
            specials.update(re.findall(r"\[[A-Z_]+\]", t))

    return list(specials)


# -----------------------------------------------------------
# 3. Prepare augmented texts
# -----------------------------------------------------------
def prepare_texts_with_special_tokens(texts, use_ner_tags=False):
    specials = extract_special_tokens(texts, use_ner_tags)
    augmented = texts + [" ".join(specials)]  # Append special tokens so TF-IDF learns them
    return augmented, specials


# -----------------------------------------------------------
# 4. Fit a TF-IDF vectorizer
# -----------------------------------------------------------
def fit_tfidf_vectorizer(
    texts,
    mode="ml",
    use_ner_tags=False,
    max_features=50000,
    min_df=3,
    max_df=0.9,
    save_path=None
):
    if mode == "transformer":
        raise ValueError("TF-IDF should NOT be used in transformer mode.")

    # A — augment data so NOT_ tokens remain
    augmented_texts, specials = prepare_texts_with_special_tokens(
        texts, use_ner_tags
    )

    # B — build vectorizer
    vectorizer = TfidfVectorizer(
        tokenizer=lambda t: custom_tokenizer(t, mode="ml", use_ner_tags=use_ner_tags),
        lowercase=False,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        analyzer="word"  # word-level tokens
    )

    # C — fit
    X = vectorizer.fit_transform(augmented_texts)

    # D — save if needed
    if save_path:
        joblib.dump(vectorizer, save_path)

    print(f"[OK] TF-IDF fitted. Vocabulary size = {len(vectorizer.vocabulary_)}")
    return vectorizer, X


# -----------------------------------------------------------
# 5. Load vectorizer
# -----------------------------------------------------------
def load_vectorizer(path):
    return joblib.load(path)


# -----------------------------------------------------------
# 6. Vectorize new text
# -----------------------------------------------------------
def vectorize_new_text(vectorizer, texts, mode="ml", use_ner_tags=False):
    if mode == "transformer":
        raise ValueError("Transformer mode should NOT use TF-IDF.")
    return vectorizer.transform(texts)
