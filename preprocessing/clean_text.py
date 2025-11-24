from IPython.utils.text import string
from preprocessing.text_utils import fix_missing_apostrophes, expand_contractions_and_slang, replace_emojis_and_emoticons, handle_negations
from preprocessing.text_utils import STOPWORDS, KEEP_WORDS
import re
import nltk
import string
from nltk.tokenize import word_tokenize

# ---------------------------
# Full Clean Text Function
# ---------------------------
def clean_text(text: str, mode="ml", use_ner_tags=False):
    """Full preprocessing pipeline"""
    if not isinstance(text, str):
        return ""

    # Remove mentions (@username) including trailing punctuation before stripping symbols
    text = re.sub(r'@\S+', '', text).strip()

    # Remove URLs (covers https, www)
    text = re.sub(r"http\S+|www\S+", "", text)

    text = text.lower().strip()

    # Replace emojis and emoticons
    text = replace_emojis_and_emoticons(text, mode=mode, use_ner_tags=use_ner_tags)

    # Fix missing apostrophes first
    text = fix_missing_apostrophes(text)

    # Remove punctuation except apostrophes and underscores
    text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "").replace("_", "")))

    # Expand contractions and slang
    text = expand_contractions_and_slang(text)

    # Remove specific word 'href'
    text = re.sub(r'\bhref\b', '', text)

    # Remove # but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # Normalize extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Negation handling
    tokens = handle_negations(tokens)

    KEEP_WORDS = KEEP_WORDS - "not" - "no"
    # Remove stopwords for ML only
    if mode == "ml":
        tokens = [t for t in tokens if t not in STOPWORDS or t in KEEP_WORDS]

    # Transformers mode: remove NOT_ prefixes
    if mode == "transformer":
        tokens = [t.replace("NOT_", "") for t in tokens]

    return " ".join(tokens)
