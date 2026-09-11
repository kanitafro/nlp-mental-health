# bert/xai/run_multilabel.py

from pathlib import Path
import numpy as np

from explain_multilabel import (
    LABELS,
    EmotionPredictor,
    IntegratedGradientsExplainer,
    create_explainer,
    load_model,
)

from utils_multilabel import (
    load_test_data,
    predict_test_set,
    compute_per_label_metrics,
    save_predictions,
    save_per_label_metrics,
    select_true_positives,
    select_false_positives,
    select_false_negatives,
    save_example_metadata,
    explain_example,
    OUTPUT_DIR,
    TEXT_COLUMN,
)

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT_ROOT / "bert" / "saved_models" / "finetuned_model_v2_7_1_2_3" / "best_model.pt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Configuration
# ============================================================
DARK_MODE = False
MAX_LENGTH = 128
N_PER_LABEL = 2               # number of examples per category per label
SHAP_MAX_EVALS = 500
IG_STEPS = 50
RANDOM_SEED = 42
THRESHOLD = 0.5

# Priority emotions (focus analysis on these)
PRIORITY_EMOTIONS = ["love", "surprise", "neutral", "disgust"]

# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTI‑LABEL XAI ANALYSIS (GoEmotions 28 classes)")
    print("=" * 70)
    print(f"\nCheckpoint: {CHECKPOINT}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Load model
    model, tokenizer, device = load_model(str(CHECKPOINT), num_labels=len(LABELS))
    predictor = EmotionPredictor(model, tokenizer, device, max_length=MAX_LENGTH)

    # Create explainers
    print("\nCreating SHAP explainer...")
    shap_explainer = create_explainer(predictor)
    print("Creating Integrated Gradients explainer...")
    ig_explainer = IntegratedGradientsExplainer(model, tokenizer, device, max_length=MAX_LENGTH)

    # Load test data
    df = load_test_data()

    # Predict
    predictions = predict_test_set(predictor, df, threshold=THRESHOLD)
    save_predictions(predictions, OUTPUT_DIR)

    # Per‑label metrics
    metrics = compute_per_label_metrics(predictions)
    metrics_df = save_per_label_metrics(metrics, OUTPUT_DIR)
    print("\nPer‑label F1 scores (top 5 / bottom 5):")
    print(metrics_df.sort_values("f1", ascending=False).head(5))
    print(metrics_df.sort_values("f1").head(5))

    # ============================================================
    # Select examples
    # ============================================================
    selected_examples = []

    # Determine which labels to analyse: all, but prioritise those with low F1 + priority list
    # We'll analyse all labels, but also select extra for priority.
    analysis_labels = LABELS

    # For each label, select true positives, false positives, false negatives
    for emotion in analysis_labels:
        tp_ex = select_true_positives(predictions, emotion, n=N_PER_LABEL)
        fp_ex = select_false_positives(predictions, emotion, n=N_PER_LABEL)
        fn_ex = select_false_negatives(predictions, emotion, n=N_PER_LABEL)

        # True positives
        for rank, (_, row) in enumerate(tp_ex.iterrows(), 1):
            example_id = f"tp_{emotion}_{rank:02d}"
            selected_examples.append({
                "example_id": example_id,
                "category": "true_positive",
                "emotion": emotion,
                "confidence": float(row[f"prob_{emotion}"]),
                "text": row[TEXT_COLUMN],
            })

        # False positives
        for rank, (_, row) in enumerate(fp_ex.iterrows(), 1):
            example_id = f"fp_{emotion}_{rank:02d}"
            selected_examples.append({
                "example_id": example_id,
                "category": "false_positive",
                "emotion": emotion,
                "confidence": float(row[f"prob_{emotion}"]),
                "text": row[TEXT_COLUMN],
            })

        # False negatives
        for rank, (_, row) in enumerate(fn_ex.iterrows(), 1):
            example_id = f"fn_{emotion}_{rank:02d}"
            selected_examples.append({
                "example_id": example_id,
                "category": "false_negative",
                "emotion": emotion,
                "confidence": float(row[f"prob_{emotion}"]),  # true probability (missed)
                "text": row[TEXT_COLUMN],
            })

    # Also sample additional examples for priority emotions (maybe more)
    for emotion in PRIORITY_EMOTIONS:
        if emotion not in analysis_labels:
            continue
        # Add extra examples (e.g., 2 more of each type)
        for category, selector in [("true_positive", select_true_positives),
                                   ("false_positive", select_false_positives),
                                   ("false_negative", select_false_negatives)]:
            extra = selector(predictions, emotion, n=2)  # extra 2
            for rank, (_, row) in enumerate(extra.iterrows(), 1):
                example_id = f"priority_{category}_{emotion}_{rank:02d}"
                selected_examples.append({
                    "example_id": example_id,
                    "category": category,
                    "emotion": emotion,
                    "confidence": float(row[f"prob_{emotion}"]),
                    "text": row[TEXT_COLUMN],
                })

    # Save metadata
    save_example_metadata(selected_examples, OUTPUT_DIR)

    # ============================================================
    # Generate explanations for each selected example
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING XAI EXPLANATIONS")
    print("=" * 70)

    for example in selected_examples:
        emotion = example["emotion"]
        category = example["category"]

        # Find the corresponding row in predictions (by text)
        matches = predictions[predictions[TEXT_COLUMN] == example["text"]]
        if len(matches) == 0:
            print(f"WARNING: Could not find row for example {example['example_id']}")
            continue
        row = matches.iloc[0]

        # Determine which emotion to explain:
        # - For true positives: explain the emotion (it's correct)
        # - For false positives: explain the predicted (false) emotion
        # - For false negatives: explain the true (missed) emotion
        # We only need one index per example.
        emotion_idx = LABELS.index(emotion)

        example_dir = OUTPUT_DIR / category / example["example_id"]

        explain_example(
            predictor=predictor,
            shap_explainer=shap_explainer,
            ig_explainer=ig_explainer,
            row=row,
            example_dir=example_dir,
            explanation_emotions=[emotion_idx],
        )

    # Final summary
    print("\n" + "=" * 70)
    print("XAI COMPLETE")
    print("=" * 70)
    print(f"Total test examples: {len(predictions):,}")
    print(f"Selected XAI examples: {len(selected_examples)}")
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()