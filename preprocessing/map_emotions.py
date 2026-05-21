"""
Emotion mapping utilities:
- Map 27 (+neutral) GoEmotion labels → 6 core emotions
- Build reverse mapping
- Apply mapping to datasets or lexicon
"""

import pandas as pd

# ---------------------------------------------------------
# 1. Define 6/7 → 27 mapping
# ---------------------------------------------------------

EMOTIONS_6 = {
    "love": ["admiration", "caring", "gratitude", "love", "neutral"],
    "joy": ["amusement", "desire", "joy", "optimism", "pride", "relief", "neutral"],
    "sadness": ["disappointment", "grief", "remorse", "sadness", "neutral"],
    "anger": ["anger", "annoyance", "disapproval", "disgust", "neutral"],
    "surprise": ["confusion", "curiosity", "excitement", "realization", "surprise", "neutral"],
    "fear": ["embarrassment", "fear", "nervousness", "neutral"]
}

EMOTIONS_7 = {
    "love": ["admiration", "caring", "gratitude", "love", "neutral"],
    "joy": ["amusement", "desire", "excitement", "joy", "optimism", "pride", "relief", "neutral"],
    "sadness": ["grief", "remorse", "sadness", "neutral"],
    "anger": ["anger", "annoyance", "disapproval", "neutral"],
    "disgust": ["disgust", "disappointment", "embarrassment", "neutral"],
    "surprise": ["confusion", "curiosity", "realization", "surprise", "neutral"],
    "fear": ["fear", "nervousness", "neutral"]
}
# ---------------------------------------------------------
# 2. Reverse mapping: 27 → 6/7
# ---------------------------------------------------------

def build_reverse_mapping(emotions_6_7: dict) -> dict:
    reverse = {}
    for core, subs in emotions_6_7.items():
        for emo in subs:
            reverse[emo] = core
    return reverse

REVERSE_MAP_6 = build_reverse_mapping(EMOTIONS_6)
REVERSE_MAP_7 = build_reverse_mapping(EMOTIONS_7)
# enforce neutral→neutral
REVERSE_MAP_6["neutral"] = "neutral"
REVERSE_MAP_7["neutral"] = "neutral"
# ---------------------------------------------------------
# 3. Apply mapping to a list of emotions
# ---------------------------------------------------------

def map_emotions_list(emotions: list, reverse_map: dict) -> list:
    """Map a list of 27 emotion labels to 6/7 core emotions."""
    if not isinstance(emotions, list):
        return []
    return list({reverse_map.get(e.lower(), e.lower()) for e in emotions})

# ---------------------------------------------------------
# 4. Apply mapping to a Pandas column
# ---------------------------------------------------------

def map_emotions_df_column(df: pd.DataFrame, col_name: str, new_col: str = "emotion_6"):
    """
    df[col_name] must contain 27-class labels.
    Returns df with a new col containing 6-core emotions.
    """
    df[new_col] = df[col_name].apply(lambda x: REVERSE_MAP_6.get(str(x).lower(), x))
    return df
