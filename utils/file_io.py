import os
import json
import csv
import pandas as pd

def save_dict_as_json(data, target_directory, filename="lexicon.json"):
    """
    Saves any Python dictionary as a JSON file.
    """
    os.makedirs(target_directory, exist_ok=True)

    filepath = os.path.join(target_directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"JSON file saved → {filepath}")
    return filepath


def save_dict_as_csv(data, target_directory, filename="lexicon.csv"):
    """
    Flattens theme/subtheme/keyword/emotions dict and saves as CSV.
    """
    os.makedirs(target_directory, exist_ok=True)

    flattened_data = []

    for category, subthemes in data.items():
        for subtheme, details in subthemes.items():

            keywords = details.get('keywords', [])
            emotions = details.get('emotions', [])

            emotions_str = ", ".join(emotions)

            if keywords:
                for kw in keywords:
                    flattened_data.append({
                        "Theme": category,
                        "Subtheme": subtheme,
                        "Keyword": kw,
                        "Emotion": emotions_str
                    })
            else:
                flattened_data.append({
                    "Theme": category,
                    "Subtheme": subtheme,
                    "Keyword": "",
                    "Emotion": emotions_str
                })

    df = pd.DataFrame(flattened_data)

    filepath = os.path.join(target_directory, filename)
    df.to_csv(filepath, index=False, encoding="utf-8")

    print(f"CSV file saved → {filepath}")
    return filepath
