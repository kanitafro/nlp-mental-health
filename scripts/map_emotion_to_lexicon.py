# run 'python scripts/map_emotions_to_lexicon.py' on terminal ONCE


"""
This script:
1. Loads the raw lexicon (27 emotions)
2. Cleans keywords + emotion labels
3. Maps the 27 emotions → 6 core emotions
4. Saves:
    - lexicon_clean_6.json
    - lexicon_clean_6.csv
"""

import os
import json
import pandas as pd

from data.lexicon.lexicon_raw import get_lexicon
from preprocessing.clean_lexicon import clean_lexicon
from preprocessing.map_emotions import REVERSE_MAP

# ------------------------------------------------------------------
# Convert a list of 27 emotions to 6-family emotions
# ------------------------------------------------------------------

def map_emotions_list_6(emotions: list):
    """Map emotions list to 6-core families."""
    if not isinstance(emotions, list):
        return []
    mapped = set()
    for emo in emotions:
        emo = emo.lower()
        mapped.add(REVERSE_MAP.get(emo, emo))
    return list(mapped)


# ------------------------------------------------------------------
# Apply mapping to entire cleaned lexicon dictionary
# ------------------------------------------------------------------

def map_lexicon_emotions_to_6(cleaned_dict):
    """Replace each subtheme's emotion list with 6-core emotion list."""
    output = {}

    for theme, subthemes in cleaned_dict.items():
        output[theme] = {}

        for subtheme, data in subthemes.items():
            keywords = data.get("keywords", [])
            emo_list_27 = data.get("emotions", [])

            emo_list_6 = map_emotions_list_6(emo_list_27)

            output[theme][subtheme] = {
                "keywords": keywords,
                "emotions": emo_list_6
            }

    return output


# ------------------------------------------------------------------
# Flatten dictionary → CSV
# ------------------------------------------------------------------

def lexicon_to_dataframe(lexicon_dict):
    rows = []
    for theme, subthemes in lexicon_dict.items():
        for subtheme, data in subthemes.items():
            keywords = data.get("keywords", [])
            emotions = data.get("emotions", [])

            for kw in keywords:
                rows.append({
                    "Theme": theme,
                    "Subtheme": subtheme,
                    "Keyword": kw,
                    "Emotion_6": ", ".join(emotions)
                })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_lexicon_emotion_mapping():
    lexicon_raw = get_lexicon()

    print("🔹 Cleaning keywords + 27 emotions...")
    cleaned_dict, df_clean = clean_lexicon(lexicon_raw)

    print("🔹 Mapping 27 → 6 emotions...")
    lexicon_6 = map_lexicon_emotions_to_6(cleaned_dict)

    print("🔹 Converting to DataFrame...")
    df_6 = lexicon_to_dataframe(lexicon_6)

    # save paths
    out_json = "data/lexicon/lexicon_clean_6.json"
    out_csv = "data/lexicon/lexicon_clean_6.csv"

    print("🔹 Saving JSON:", out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(lexicon_6, f, ensure_ascii=False, indent=2)

    print("🔹 Saving CSV:", out_csv)
    df_6.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n✅ DONE! Emotion-mapped lexicon saved.")
    print("   - lexicon_clean_6.json")
    print("   - lexicon_clean_6.csv")

if __name__ == "__main__":
    run_lexicon_emotion_mapping()
