import pandas as pd
import json
import re
from preprocessing.text_utils import MISSING_APOSTROPHE_MAP, CONTRACTION_MAP
from preprocessing.text_utils import fix_missing_apostrophes

# --- helper functions ---------------------------------------------------------

def expand_contractions(text: str) -> str:
    """Expand contractions"""
    tokens = text.split()
    expanded = []
    for token in tokens:
        lower = token.lower()
        if lower in CONTRACTION_MAP:
            expanded.extend(CONTRACTION_MAP[lower].split())
        else:
            expanded.append(token)
    return " ".join(expanded)


def clean_keyword(text):
    """Lowercase, remove apostrophes/quotes, keep letters/numbers/hyphens."""
    if not isinstance(text, str):
        return text
    text = text.lower().strip()
    text = re.sub(r"[“”\"'’]", "", text)     # remove apostrophes/quotes
    text = re.sub(r"[^a-z0-9\s-]", "", text) # keep only a–z 0–9 hyphens
    text = re.sub(r"\s+", " ", text)         # collapse spaces
    text = fix_missing_apostrophes(text)
    text = expand_contractions(text)
    return text

def clean_emotions_list(emotions):
    """Convert list → lowercase list without duplicates."""
    if not isinstance(emotions, list):
        return []
    cleaned = [e.strip().lower() for e in emotions if isinstance(e, str)]
    return list(dict.fromkeys(cleaned))      # remove duplicates while preserving order

# --- main cleaning function ---------------------------------------------------

def clean_lexicon(lexicon_dict):
    """
    lexicon_dict format:

    {
        "Theme": {
            "Subtheme": {
                "keywords": [...],
                "emotions": [...]
            }
        }
    }

    Returns:
        cleaned_dict (same structure but cleaned)
        df_clean (flattened DataFrame with cleaned keywords/emotions)
    """

    cleaned_dict = {}
    rows = []

    for theme, subthemes in lexicon_dict.items():
        cleaned_dict.setdefault(theme, {})

        for subtheme, data in subthemes.items():
            keywords = data.get("keywords", [])
            emotions = data.get("emotions", [])

            # clean keywords + emotions
            cleaned_keywords = [clean_keyword(k) for k in keywords]
            cleaned_emotions = clean_emotions_list(emotions)

            cleaned_dict[theme][subtheme] = {
                "keywords": cleaned_keywords,
                "emotions": cleaned_emotions
            }

            # also build rows for DataFrame
            for kw in cleaned_keywords:
                rows.append({
                    "Theme": theme,
                    "Subtheme": subtheme,
                    "Keyword": kw,
                    "Emotion": ", ".join(cleaned_emotions)
                })

    df = pd.DataFrame(rows)
    return cleaned_dict, df
