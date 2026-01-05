import re
import joblib
import math
import numpy as np
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
# 2. Create a pickle-able tokenizer class
# -----------------------------------------------------------
class CustomTokenizer:
    def __init__(self, use_ner_tags=False):
        self.use_ner_tags = use_ner_tags
    
    def __call__(self, text):
        return custom_tokenizer(text, mode="ml", use_ner_tags=self.use_ner_tags)


# -----------------------------------------------------------
# 3. Extract all NOT_ and NER tokens → force-keep in vocabulary
# -----------------------------------------------------------
def extract_special_tokens(texts, use_ner_tags=False):
    specials = set()

    for t in texts:
        # Skip None and NaN values
        if t is None or (isinstance(t, float) and np.isnan(t)):
            continue
            
        # Convert to string if not already
        text_str = str(t)
        
        specials.update(re.findall(r"NOT_\w+", text_str))
        if use_ner_tags:
            specials.update(re.findall(r"\[[A-Z_]+\]", text_str))

    return list(specials)


# -----------------------------------------------------------
# 4. Prepare augmented texts
# -----------------------------------------------------------
def prepare_texts_with_special_tokens(texts, use_ner_tags=False):
    # Clean the texts first - remove NaN/None values and convert to strings
    cleaned_texts = []
    for t in texts:
        if t is None or (isinstance(t, float) and np.isnan(t)):
            cleaned_texts.append("")  # Replace with empty string
        else:
            cleaned_texts.append(str(t))
    
    texts = cleaned_texts
    
    specials = extract_special_tokens(texts, use_ner_tags)
    augmented = texts + [" ".join(specials)]  # Append special tokens so TF-IDF learns them
    return augmented, specials


# -----------------------------------------------------------
# 5. Fit a TF-IDF vectorizer
# -----------------------------------------------------------
def fit_tfidf_vectorizer(
    texts,
    mode="ml",
    use_ner_tags=False,
    max_features=50000,
    min_df=2,
    max_df=0.9,
    save_path=None
):
    if mode == "transformer":
        raise ValueError("TF-IDF should NOT be used in transformer mode.")

    # A — augment data so NOT_ tokens remain
    augmented_texts, specials = prepare_texts_with_special_tokens(texts, use_ner_tags)

    # B — build vectorizer with pickle-able tokenizer
    vectorizer = TfidfVectorizer(
        tokenizer=CustomTokenizer(use_ner_tags=use_ner_tags),  # Use class instead of function
        lowercase=False,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        analyzer="word"  # word-level tokens
    )

    # C — fit on augmented texts to learn all vocabulary, including special tokens
    vectorizer.fit(augmented_texts)

    # Now, transform only the original texts to get X with the correct number of samples
    X = vectorizer.transform(texts)

    # D — save if needed
    if save_path:
        joblib.dump(vectorizer, save_path)

    print(f"[OK] TF-IDF fitted. Vocabulary size = {len(vectorizer.vocabulary_)}")
    print("If there's a warning about token_pattern it means custom tokenizer is working! All good!")
    return vectorizer, X


# -----------------------------------------------------------
# 6. Load vectorizer
# -----------------------------------------------------------
def load_vectorizer(path):
    return joblib.load(path)


# -----------------------------------------------------------
# 7. Vectorize new text
# -----------------------------------------------------------
def vectorize_new_text(vectorizer, texts, mode="ml", use_ner_tags=False):
    if mode == "transformer":
        raise ValueError("Transformer mode should NOT use TF-IDF.")
    return vectorizer.transform(texts)
