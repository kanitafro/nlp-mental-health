# bert/XAI/run_xai.py

from pathlib import Path
#import json
#import csv

import numpy as np
#import pandas as pd
#from sklearn.metrics import confusion_matrix

from explain import (
    LABELS,
    EmotionPredictor,
    IntegratedGradientsExplainer,
    create_explainer,
    load_model,
)

from utils_xai import (
    clean_label,
    load_test_data,
    predict_test_set,
    save_predictions,
    save_confusion_summary,
    select_true_positives,
    select_false_positives,
    select_confusion_examples,
    save_example_metadata,
    explain_example,
)

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    PROJECT_ROOT
    / "bert"
    / "checkpoints_v2_7"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "bert"
    / "xai"
    / "outputs"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

TEXT_COLUMN = "clean_text_transf"
LABEL_COLUMN = "label"

# ============================================================
# XAI configuration
# ============================================================

DARK_MODE = False

MAX_LENGTH = 128

# Number of representative examples per emotion
N_TRUE_POSITIVES = 2
N_FALSE_POSITIVES = 2

# Number of examples per important confusion pair
N_CONFUSION_EXAMPLES = 2

# SHAP computation
SHAP_MAX_EVALS = 500

# Integrated Gradients
IG_STEPS = 50

# Random seed
RANDOM_SEED = 42


# ============================================================
# Special attention emotions
# ============================================================

# These emotions deserve additional analysis because of
# their observed precision/recall behavior and semantic
# complexity.

PRIORITY_EMOTIONS = [
    "love",
    "surprise",
]


# ============================================================
# Example texts
# ============================================================

TEXTS = [
    "I am so happy that everything worked out perfectly.",
    "I feel completely heartbroken and unable to stop crying.",
    "I am furious about what happened to me.",
    "I absolutely adore this person and feel so much love for them.",
    "I cannot believe what just happened.",
    "That news was terrifying and made me feel unsafe.",
    "The situation was disgusting and made me feel sick.",
]



# ============================================================
# Main XAI analysis
# ============================================================

def main():

    np.random.seed(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "PHASE 1 XAI ANALYSIS"
    )
    print(
        "DistilBERT - 7 Emotion Single-Label Classification"
    )
    print("=" * 70)

    print(
        f"\nCheckpoint:"
        f"\n{CHECKPOINT}"
    )

    print(
        f"\nOutput directory:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        f"\nDark mode: {DARK_MODE}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, tokenizer, device = load_model(
        str(CHECKPOINT)
    )

    print(
        f"\nDevice: {device}"
    )

    predictor = EmotionPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=MAX_LENGTH,
    )

    # --------------------------------------------------------
    # Create SHAP
    # --------------------------------------------------------

    print(
        "\nCreating SHAP explainer..."
    )

    shap_explainer = create_explainer(
        predictor
    )

    # --------------------------------------------------------
    # Create IG
    # --------------------------------------------------------

    print(
        "Creating Integrated Gradients explainer..."
    )

    ig_explainer = (
        IntegratedGradientsExplainer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=MAX_LENGTH,
        )
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    df = load_test_data()

    # --------------------------------------------------------
    # Predict entire test set
    # --------------------------------------------------------

    predictions = predict_test_set(
        predictor,
        df,
    )

    save_predictions(
        predictions
    )

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    accuracy = (
        predictions["correct"]
        .mean()
    )

    print(
        f"\nTest accuracy: "
        f"{accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Confusion pairs
    # --------------------------------------------------------

    confusion_pairs = (
        save_confusion_summary(
            predictions
        )
    )

    print(
        "\nTop confusion pairs:"
    )

    for pair in confusion_pairs[:10]:

        print(
            f"  {pair['true_emotion']:10s}"
            f" -> "
            f"{pair['predicted_emotion']:10s}"
            f" : {pair['count']}"
        )

    # --------------------------------------------------------
    # Selected examples
    # --------------------------------------------------------

    selected_examples = []

    # ========================================================
    # A. TRUE POSITIVES
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SELECTING TRUE POSITIVES"
    )
    print("=" * 70)

    for emotion in LABELS:

        examples = select_true_positives(
            predictions,
            emotion,
            n=N_TRUE_POSITIVES,
        )

        print(
            f"\n{emotion.upper()}: "
            f"{len(examples)} examples"
        )

        for rank, (_, row) in enumerate(
            examples.iterrows(),
            start=1,
        ):

            example_id = (
                f"tp_{emotion}_{rank:02d}"
            )

            selected_examples.append(
                {
                    "example_id": example_id,
                    "category": "true_positive",
                    "true_emotion": emotion,
                    "predicted_emotion": emotion,
                    "confidence": float(
                        row[
                            "predicted_probability"
                        ]
                    ),
                    "text": row[
                        TEXT_COLUMN
                    ],
                }
            )

    # ========================================================
    # B. FALSE POSITIVES
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SELECTING FALSE POSITIVES"
    )
    print("=" * 70)

    for emotion in LABELS:

        examples = select_false_positives(
            predictions,
            emotion,
            n=N_FALSE_POSITIVES,
        )

        print(
            f"\n{emotion.upper()}: "
            f"{len(examples)} examples"
        )

        for rank, (_, row) in enumerate(
            examples.iterrows(),
            start=1,
        ):

            true_emotion = clean_label(
                row[LABEL_COLUMN]
            )

            example_id = (
                f"fp_{emotion}_"
                f"{true_emotion}_"
                f"{rank:02d}"
            )

            selected_examples.append(
                {
                    "example_id": example_id,
                    "category": "false_positive",
                    "true_emotion": true_emotion,
                    "predicted_emotion": emotion,
                    "confidence": float(
                        row[
                            "predicted_probability"
                        ]
                    ),
                    "text": row[
                        TEXT_COLUMN
                    ],
                }
            )

    # ========================================================
    # C. IMPORTANT CONFUSION PAIRS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "SELECTING IMPORTANT CONFUSION CASES"
    )
    print("=" * 70)

    # Always prioritize confusion involving
    # love or surprise.

    priority_pairs = []
    normal_pairs = []
    
    for pair in confusion_pairs:
    
        involves_priority = (
            pair["true_emotion"] in PRIORITY_EMOTIONS
            or
            pair["predicted_emotion"] in PRIORITY_EMOTIONS
        )
    
        if involves_priority:
            priority_pairs.append(pair)
        else:
            normal_pairs.append(pair)
    
    # --------------------------------------------------------
    # Balanced confusion-pair selection
    # --------------------------------------------------------
    
    selected_pairs = []
    
    # First: globally most frequent confusion pairs
    selected_pairs.extend(
        normal_pairs[:5]
    )
    
    # Then: most important priority-emotion confusions
    selected_pairs.extend(
        priority_pairs[:5]
    )
    
    # Remove duplicates while preserving order
    seen_pairs = set()
    unique_pairs = []
    
    for pair in selected_pairs:
    
        key = (
            pair["true_emotion"],
            pair["predicted_emotion"],
        )
    
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append(pair)
    
    selected_pairs = unique_pairs[:10]

    for pair_index, pair in enumerate(
        selected_pairs,
        start=1,
    ):

        true_emotion = pair[
            "true_emotion"
        ]

        predicted_emotion = pair[
            "predicted_emotion"
        ]

        examples = (
            select_confusion_examples(
                predictions,
                true_emotion=true_emotion,
                predicted_emotion=predicted_emotion,
                n=N_CONFUSION_EXAMPLES,
            )
        )

        print(
            f"\n{true_emotion.upper()}"
            f" -> "
            f"{predicted_emotion.upper()}"
            f" ({pair['count']} cases)"
        )

        for rank, (_, row) in enumerate(
            examples.iterrows(),
            start=1,
        ):

            example_id = (
                f"confusion_"
                f"{true_emotion}_"
                f"{predicted_emotion}_"
                f"{rank:02d}"
            )

            selected_examples.append(
                {
                    "example_id": example_id,
                    "category": "confusion",
                    "true_emotion": true_emotion,
                    "predicted_emotion": predicted_emotion,
                    "confidence": float(
                        row[
                            "predicted_probability"
                        ]
                    ),
                    "text": row[
                        TEXT_COLUMN
                    ],
                }
            )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_example_metadata(
        selected_examples
    )

    # --------------------------------------------------------
    # Explain selected examples
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "GENERATING XAI EXPLANATIONS"
    )
    print("=" * 70)

    for example in selected_examples:

        category = example[
            "category"
        ]

        true_emotion = example[
            "true_emotion"
        ]

        predicted_emotion = example[
            "predicted_emotion"
        ]

        # Find exact test row
        matches = predictions[
            (
                predictions[TEXT_COLUMN]
                == example["text"]
            )
            &
            (
                predictions[LABEL_COLUMN]
                == true_emotion
            )
            &
            (
                predictions[
                    "predicted_label"
                ]
                == predicted_emotion
            )
        ]

        if len(matches) == 0:

            print(
                f"\nWARNING: Could not find "
                f"{example['example_id']}"
            )

            continue

        row = matches.iloc[0]

        example_dir = (
            OUTPUT_DIR
            / category
            / example["example_id"]
        )

        # ----------------------------------------------------
        # What should be explained?
        #
        # True positives:
        #   predicted emotion
        #
        # False positives/confusions:
        #   predicted emotion
        #   true emotion
        # ----------------------------------------------------

        explanation_emotions = [
            LABELS.index(
                predicted_emotion
            )
        ]

        if (
            category != "true_positive"
            and
            true_emotion
            != predicted_emotion
        ):

            explanation_emotions.append(
                LABELS.index(
                    true_emotion
                )
            )

        # Remove duplicates while preserving order
        explanation_emotions = list(
            dict.fromkeys(
                explanation_emotions
            )
        )

        explain_example(
            predictor=predictor,
            shap_explainer=shap_explainer,
            ig_explainer=ig_explainer,
            row=row,
            example_dir=example_dir,
            explanation_emotions=(
                explanation_emotions
            ),
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "PHASE 1 XAI COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nTotal test examples:"
        f" {len(predictions):,}"
    )

    print(
        f"Test accuracy:"
        f" {accuracy:.4f}"
    )

    print(
        f"Selected XAI examples:"
        f" {len(selected_examples)}"
    )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()