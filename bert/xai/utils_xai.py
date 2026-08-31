# bert/xai/utils_xai.py
"""
Utility functions used in run_xai.py
"""
from pathlib import Path

import pandas as pd
import numpy as np
import json

from explain import LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"
TEXT_COLUMN = "clean_text_transf"
LABEL_COLUMN = "label"
OUTPUT_DIR = PROJECT_ROOT / "bert" / "xai" / "outputs"
SHAP_MAX_EVALS = 500
IG_STEPS = 50
DARK_MODE = False


from explain import (
    explain_text,
    save_token_contributions,
    save_shap_html,
    get_top_tokens,
    save_ig_token_contributions,
    save_ig_html,
)

def clean_label(label):
    """
    Normalize label representation.
    """

    return str(label).strip().lower()


def load_test_data():
    """
    Load the held-out test set.

    Expected columns:
        text
        label
    """

    print("\nLoading test dataset...")
    print(f"Test CSV: {TEST_CSV}")

    if not TEST_CSV.exists():

        raise FileNotFoundError(
            f"\nTest dataset was not found:\n"
            f"{TEST_CSV}\n\n"
            f"Set TEST_CSV in utils_xai.py to the "
            f"actual held-out Phase 1 test CSV."
        )

    df = pd.read_csv(
        TEST_CSV
    )

    if TEXT_COLUMN not in df.columns:

        raise ValueError(
            f"Text column '{TEXT_COLUMN}' "
            f"was not found.\n"
            f"Available columns: "
            f"{list(df.columns)}"
        )

    if LABEL_COLUMN not in df.columns:

        raise ValueError(
            f"Label column '{LABEL_COLUMN}' "
            f"was not found.\n"
            f"Available columns: "
            f"{list(df.columns)}"
        )

    df = df[
        [
            TEXT_COLUMN,
            LABEL_COLUMN,
        ]
    ].copy()

    df = df.dropna()

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .astype(str)
        .str.strip()
    )

    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .apply(clean_label)
    )

    # Keep only the seven Phase 1 emotions
    df = df[
        df[LABEL_COLUMN].isin(LABELS)
    ].reset_index(drop=True)

    if len(df) == 0:

        raise ValueError(
            "No valid Phase 1 emotion examples "
            "were found in the test dataset."
        )

    print(
        f"Loaded {len(df):,} test examples."
    )

    print("\nTest label distribution:")

    print(
        df[LABEL_COLUMN]
        .value_counts()
        .reindex(LABELS)
        .fillna(0)
        .astype(int)
        .to_string()
    )

    return df


# ============================================================
# Run model over entire test set
# ============================================================

def predict_test_set(
    predictor,
    df,
):
    """
    Generate predictions for every test example.
    """

    texts = df[
        TEXT_COLUMN
    ].tolist()

    print(
        "\nGenerating model predictions..."
    )

    probabilities = predictor(
        texts
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1,
    )

    predicted_labels = [
        LABELS[index]
        for index in predicted_indices
    ]

    predicted_probabilities = (
        probabilities[
            np.arange(len(df)),
            predicted_indices,
        ]
    )

    result = df.copy()

    result["predicted_label"] = (
        predicted_labels
    )

    result["predicted_probability"] = (
        predicted_probabilities
    )

    result["correct"] = (
        result[LABEL_COLUMN]
        == result["predicted_label"]
    )

    # Save probability for every emotion
    for index, emotion in enumerate(
        LABELS
    ):

        result[
            f"prob_{emotion}"
        ] = probabilities[:, index]

    return result


# ============================================================
# Select representative examples
# ============================================================

def select_true_positives(
    predictions,
    emotion,
    n=2,
):
    """
    Select high-confidence true positives.

    These are examples where:

        true label == predicted label == emotion

    Highest-confidence examples are selected.
    """

    candidates = predictions[
        (predictions[LABEL_COLUMN] == emotion)
        &
        (predictions["predicted_label"] == emotion)
    ].copy()

    candidates = candidates.sort_values(
        "predicted_probability",
        ascending=False,
    )

    return candidates.head(n)


def select_false_positives(
    predictions,
    predicted_emotion,
    n=2,
):
    """
    Select high-confidence false positives.

    These are examples where:

        predicted label == predicted_emotion
        true label != predicted_emotion

    Highest-confidence mistakes are selected.
    """

    candidates = predictions[
        (predictions["predicted_label"] == predicted_emotion)
        &
        (predictions[LABEL_COLUMN] != predicted_emotion)
    ].copy()

    candidates = candidates.sort_values(
        "predicted_probability",
        ascending=False,
    )

    return candidates.head(n)


def get_confusion_pairs(
    predictions,
):
    """
    Return confusion pairs sorted by frequency.

    A pair is:

        true emotion -> predicted emotion
    """

    pairs = []

    for true_emotion in LABELS:

        for predicted_emotion in LABELS:

            if true_emotion == predicted_emotion:
                continue

            count = len(
                predictions[
                    (predictions[LABEL_COLUMN] == true_emotion)
                    &
                    (
                        predictions["predicted_label"]
                        == predicted_emotion
                    )
                ]
            )

            if count > 0:

                pairs.append(
                    {
                        "true_emotion": true_emotion,
                        "predicted_emotion": predicted_emotion,
                        "count": count,
                    }
                )

    pairs.sort(
        key=lambda x: x["count"],
        reverse=True,
    )

    return pairs


def select_confusion_examples(
    predictions,
    true_emotion,
    predicted_emotion,
    n=2,
):
    """
    Select high-confidence examples for a
    particular confusion pair.
    """

    candidates = predictions[
        (predictions[LABEL_COLUMN] == true_emotion)
        &
        (
            predictions["predicted_label"]
            == predicted_emotion
        )
    ].copy()

    candidates = candidates.sort_values(
        "predicted_probability",
        ascending=False,
    )

    return candidates.head(n)


# ============================================================
# Save prediction summary
# ============================================================

def save_predictions(
    predictions,
):
    """
    Save complete test-set predictions.
    """

    path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    predictions.to_csv(
        path,
        index=False,
    )

    print(
        f"\nSaved test predictions:"
        f"\n  {path}"
    )


# ============================================================
# Save confusion summary
# ============================================================

def save_confusion_summary(
    predictions,
):
    """
    Save confusion-pair statistics.
    """

    pairs = get_confusion_pairs(
        predictions
    )

    path = (
        OUTPUT_DIR
        / "confusion_pairs.csv"
    )

    pd.DataFrame(
        pairs
    ).to_csv(
        path,
        index=False,
    )

    return pairs


# ============================================================
# Save selected example metadata
# ============================================================

def save_example_metadata(
    examples,
):
    """
    Save metadata describing every XAI example.
    """

    path = (
        OUTPUT_DIR
        / "selected_examples.csv"
    )

    pd.DataFrame(
        examples
    ).to_csv(
        path,
        index=False,
    )

    print(
        f"\nSaved selected example metadata:"
        f"\n  {path}"
    )


# ============================================================
# Explain one example
# ============================================================

def explain_example(
    predictor,
    shap_explainer,
    ig_explainer,
    row,
    example_dir,
    explanation_emotions,
):
    """
    Generate SHAP + IG explanations for one
    selected test example.

    explanation_emotions:
        list of emotion indices to explain.
    """

    example_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = str(
        row[TEXT_COLUMN]
    )

    true_emotion = clean_label(
        row[LABEL_COLUMN]
    )

    predicted_emotion = clean_label(
        row["predicted_label"]
    )

    # --------------------------------------------------------
    # Save prediction metadata
    # --------------------------------------------------------

    prediction_info = {
        "text": text,
        "true_emotion": true_emotion,
        "predicted_emotion": predicted_emotion,
        "predicted_probability": float(
            row["predicted_probability"]
        ),
        "correct": bool(
            row["correct"]
        ),
    }

    with (
        example_dir
        / "prediction.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            prediction_info,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        f"TRUE:      {true_emotion}"
    )

    print(
        f"PREDICTED: {predicted_emotion}"
    )

    print(
        f"CONFIDENCE: "
        f"{float(row['predicted_probability']):.4f}"
    )

    print(
        f"\nTEXT:\n{text}"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print(
        "\nCalculating SHAP..."
    )

    shap_values = explain_text(
        shap_explainer,
        text,
        max_evals=SHAP_MAX_EVALS,
    )

    # Save all SHAP token values
    save_token_contributions(
        shap_values,
        str(
            example_dir
            / "shap_token_contributions.csv"
        ),
    )

    # Explain each requested emotion
    for emotion_index in explanation_emotions:

        emotion = LABELS[
            emotion_index
        ]

        # ----------------------------------------------------
        # Light mode
        # ----------------------------------------------------

        save_shap_html(
            shap_values,
            str(
                example_dir
                / f"shap_{emotion}_light.html"
            ),
            emotion_index=emotion_index,
            dark_mode=False,
        )

        # ----------------------------------------------------
        # Dark mode
        # ----------------------------------------------------

        save_shap_html(
            shap_values,
            str(
                example_dir
                / f"shap_{emotion}_dark.html"
            ),
            emotion_index=emotion_index,
            dark_mode=True,
        )

        # ----------------------------------------------------
        # Top tokens
        # ----------------------------------------------------

        top_tokens = get_top_tokens(
            shap_values,
            emotion,
            top_k=10,
        )

        top_path = (
            example_dir
            / f"shap_top_tokens_{emotion}.json"
        )

        with top_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                top_tokens,
                f,
                indent=4,
                ensure_ascii=False,
            )

    # --------------------------------------------------------
    # Integrated Gradients
    # --------------------------------------------------------

    for emotion_index in explanation_emotions:

        emotion = LABELS[
            emotion_index
        ]

        print(
            f"Calculating IG for "
            f"'{emotion}'..."
        )

        ig_result = (
            ig_explainer.explain(
                text=text,
                target_emotion_index=emotion_index,
                n_steps=IG_STEPS,
            )
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        save_ig_token_contributions(
            ig_result,
            str(
                example_dir
                / f"ig_{emotion}_tokens.csv"
            ),
        )

        # ----------------------------------------------------
        # Light mode
        # ----------------------------------------------------

        save_ig_html(
            ig_result,
            str(
                example_dir
                / f"ig_{emotion}_light.html"
            ),
            dark_mode=False,
        )

        # ----------------------------------------------------
        # Dark mode
        # ----------------------------------------------------

        save_ig_html(
            ig_result,
            str(
                example_dir
                / f"ig_{emotion}_dark.html"
            ),
            dark_mode=True,
        )

    print(
        f"\nSaved explanations to:"
        f"\n  {example_dir}"
    )