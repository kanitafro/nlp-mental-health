"""
build_lexicon.py
----------------
Builds and saves cleaned versions of the lexicon:
- Loads raw lexicon from lexicon_raw.get_lexicon()
- Cleans it using clean_lexicon()
- Saves CSV and JSON versions
"""

import os
import json
import pandas as pd

from data.lexicon.lexicon_raw import get_lexicon
from preprocessing.clean_lexicon import clean_lexicon
from utils.file_io import save_dict_as_csv, save_dict_as_json

# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def build_and_save_lexicon(output_dir="data/lexicon"):
    """
    Full pipeline:
    1. Load raw lexicon
    2. Clean lexicon
    3. Convert to DataFrame
    4. Save JSON & CSV
    """
    print("📥 Loading raw lexicon...")
    lexicon = get_lexicon()  # dictionary
    
    # --------------------------------
    # Save raw lexicon as json and csv
    # --------------------------------
    print("Getting raw lexicon:")
    json_path = os.path.join(output_dir, "lexicon.json")
    csv_path = os.path.join(output_dir, "lexicon.csv")

    print(f"💾 Saving raw JSON → {json_path}")
    save_dict_as_json(lexicon, output_dir, filename="lexicon.json")

    print(f"💾 Saving raw CSV → {csv_path}")
    save_dict_as_csv(lexicon, output_dir, filename="lexicon.csv")

    print("\n🧹 Cleaning lexicon...")
    lex_clean, df_clean = clean_lexicon(lexicon)

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------
    # Save cleaned versions
    # --------------------------
    clean_json_path = os.path.join(output_dir, "lexicon_clean.json")
    clean_csv_path = os.path.join(output_dir, "lexicon_clean.csv")

    print(f"💾 Saving cleaned JSON → {clean_json_path}")
    save_dict_as_json(lex_clean, output_dir, filename="lexicon_clean.json")

    print(f"💾 Saving cleaned CSV → {clean_csv_path}")
    save_dict_as_csv(lex_clean, output_dir, filename="lexicon_clean.csv")

    print("✅ Lexicon build complete!\n")
    return lex_clean, df_clean

