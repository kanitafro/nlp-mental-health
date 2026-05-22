import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# run 'python -m scripts.map_emotion_to_lexicon' on terminal

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
from preprocessing.map_emotions import REVERSE_MAP_6, REVERSE_MAP_7, map_emotions_list
from data.lexicon.build_lexicon import build_and_save_lexicon

# ------------------------------------------------------------------
# Convert a list of 27 emotions to 6/7-family emotions
# ------------------------------------------------------------------

def map_emotions_list_6_7(emotions: list, reverse_map: dict = REVERSE_MAP_6) -> list:
    """Map emotions list to 6/7-core families."""
    if not isinstance(emotions, list):
        return []
    mapped = set()
    for emo in emotions:
        emo = emo.lower()
        mapped.add(reverse_map.get(emo, emo))
    return list(mapped)


# ------------------------------------------------------------------
# Apply mapping to entire cleaned lexicon dictionary
# ------------------------------------------------------------------

def map_lexicon_emotions_to_6_7(cleaned_dict, reverse_map: dict = REVERSE_MAP_6) -> dict:
    """Replace each subtheme's emotion list with 6/7-core emotion list."""
    output = {}

    for theme, subthemes in cleaned_dict.items():
        output[theme] = {}

        for subtheme, data in subthemes.items():
            keywords = data.get("keywords", [])
            emo_list_27 = data.get("emotions", [])
            requires_lexical_evidence = data.get("requires_lexical_evidence", False)

            emo_list_6 = map_emotions_list_6_7(emo_list_27, reverse_map)

            output[theme][subtheme] = {
                "keywords": keywords,
                "emotions": emo_list_6,
                "requires_lexical_evidence": requires_lexical_evidence
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
            requires_lexical_evidence = data.get("requires_lexical_evidence", False)

            for kw in keywords:
                rows.append({
                    "Theme": theme,
                    "Subtheme": subtheme,
                    "Keyword": kw,
                    "Emotion_6": ", ".join(emotions),
                    "Requires_Lexical_Evidence": requires_lexical_evidence
                })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_lexicon_emotion_mapping():
    lexicon_raw = get_lexicon()

    print("🔹 Cleaning keywords + 27 emotions...")
    cleaned_dict, df_clean = clean_lexicon(lexicon_raw)

    print("🔹 Mapping 27 → 6/7 emotions...")
    lexicon_6 = map_lexicon_emotions_to_6_7(cleaned_dict, REVERSE_MAP_6)
    lexicon_7 = map_lexicon_emotions_to_6_7(cleaned_dict, REVERSE_MAP_7)

    print("🔹 Converting to DataFrame...")
    df_6 = lexicon_to_dataframe(lexicon_6)
    df_7 = lexicon_to_dataframe(lexicon_7)

    # save paths
    out6_json = "../data/lexicon/lexicon_clean_6.json"
    out7_json = "../data/lexicon/lexicon_clean_7.json"
    out6_csv = "../data/lexicon/lexicon_clean_6.csv"
    out7_csv = "../data/lexicon/lexicon_clean_7.csv"


    print("\n🔹 Saving JSON (6-core):", out6_json)
    with open(out6_json, "w", encoding="utf-8") as f:
        json.dump(lexicon_6, f, ensure_ascii=False, indent=2)

    print("🔹 Saving CSV (6-core):", out6_csv)
    df_6.to_csv(out6_csv, index=False, encoding="utf-8")

    print("\n🔹 Saving JSON (7-core):", out7_json)
    with open(out7_json, "w", encoding="utf-8") as f:
        json.dump(lexicon_7, f, ensure_ascii=False, indent=2)

    print("🔹 Saving CSV (7-core):", out7_csv)
    df_7.to_csv(out7_csv, index=False, encoding="utf-8")

    print("\n✅ DONE! Emotion-mapped lexicon saved.")
    print("   - lexicon_clean_6.json")
    print("   - lexicon_clean_6.csv")
    print("   - lexicon_clean_7.json")
    print("   - lexicon_clean_7.csv")

if __name__ == "__main__":
    build_and_save_lexicon()
    run_lexicon_emotion_mapping()
