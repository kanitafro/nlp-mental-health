# bert/xai/utils_multilabel.py

from pathlib import Path
import pandas as pd
import numpy as np
import json

from explain_multilabel import LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "goemotions" / "test_preprocessed.csv"
TEXT_COLUMN = "text"   # adjust if column name differs
LABEL_COLUMNS = LABELS              # assume binary columns exist for each emotion
OUTPUT_DIR = PROJECT_ROOT / "bert" / "xai" / "outputs_multilabel_og"
SHAP_MAX_EVALS = 500
IG_STEPS = 50
DARK_MODE = False

from explain_multilabel import (
    explain_text,
    save_token_contributions,
    save_shap_html,
    get_top_tokens,
    save_ig_token_contributions,
    save_ig_html,
)

def load_test_data():
    """Load GoEmotions test set with multi‑label columns."""
    print("\nLoading test dataset...")
    print(f"Test CSV: {TEST_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    df = pd.read_csv(TEST_CSV)

    # Ensure required columns exist
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Text column '{TEXT_COLUMN}' not found. Available: {list(df.columns)}")

    # Check which label columns exist
    existing_labels = [col for col in LABELS if col in df.columns]
    if not existing_labels:
        raise ValueError("None of the label columns found in CSV. Expected one per emotion.")
    missing = set(LABELS) - set(existing_labels)
    if missing:
        print(f"Warning: Missing label columns: {missing}. They will be filled with 0.")

    # Ensure all label columns are present (fill missing with 0)
    for label in LABELS:
        if label not in df.columns:
            df[label] = 0

    # Keep only text and label columns
    cols = [TEXT_COLUMN] + LABELS
    df = df[cols].copy()
    df = df.dropna(subset=[TEXT_COLUMN])

    # Convert label columns to binary ints (0/1)
    for label in LABELS:
        df[label] = df[label].astype(int)

    print(f"Loaded {len(df):,} test examples.")
    # Show label frequencies
    counts = df[LABELS].sum().sort_values(ascending=False)
    print("\nLabel frequencies in test set:")
    print(counts.to_string())
    return df

def predict_test_set(predictor, df, threshold=0.5):
    """Generate predictions for all test examples."""
    texts = df[TEXT_COLUMN].tolist()
    print("\nGenerating model predictions...")
    probs = predictor(texts)   # shape (n, 28)

    # Binary predictions
    pred_binary = (probs >= threshold).astype(int)

    # Store results
    result = df.copy()
    for i, label in enumerate(LABELS):
        result[f"prob_{label}"] = probs[:, i]
        result[f"pred_{label}"] = pred_binary[:, i]

    # Compute per‑label metrics (optional)
    return result

def compute_per_label_metrics(predictions):
    """Compute precision, recall, f1 per label."""
    metrics = {}
    for label in LABELS:
        true = predictions[label].values
        pred = predictions[f"pred_{label}"].values
        tp = np.sum((true == 1) & (pred == 1))
        fp = np.sum((true == 0) & (pred == 1))
        fn = np.sum((true == 1) & (pred == 0))
        tn = np.sum((true == 0) & (pred == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[label] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return metrics

def save_predictions(predictions, output_dir):
    path = output_dir / "test_predictions.csv"
    predictions.to_csv(path, index=False)
    print(f"\nSaved test predictions: {path}")

def save_per_label_metrics(metrics, output_dir):
    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.index.name = "emotion"
    path = output_dir / "per_label_metrics.csv"
    df.to_csv(path)
    print(f"Saved per‑label metrics: {path}")
    return df

# ============================================================
# Selection functions (per emotion)
# ============================================================
def select_true_positives(predictions, emotion, n=2):
    """High‑confidence examples where true=1 and pred=1 for this emotion."""
    candidates = predictions[(predictions[emotion] == 1) & (predictions[f"pred_{emotion}"] == 1)]
    candidates = candidates.sort_values(f"prob_{emotion}", ascending=False)
    return candidates.head(n)

def select_false_positives(predictions, emotion, n=2):
    """High‑confidence examples where true=0 and pred=1."""
    candidates = predictions[(predictions[emotion] == 0) & (predictions[f"pred_{emotion}"] == 1)]
    candidates = candidates.sort_values(f"prob_{emotion}", ascending=False)
    return candidates.head(n)

def select_false_negatives(predictions, emotion, n=2):
    """High‑true‑probability examples where true=1 and pred=0 (missed)."""
    candidates = predictions[(predictions[emotion] == 1) & (predictions[f"pred_{emotion}"] == 0)]
    # Sort by true probability (high confidence but missed)
    candidates = candidates.sort_values(f"prob_{emotion}", ascending=False)
    return candidates.head(n)

def save_example_metadata(examples, output_dir):
    path = output_dir / "selected_examples.csv"
    pd.DataFrame(examples).to_csv(path, index=False)
    print(f"\nSaved selected example metadata: {path}")

# ============================================================
# Explain one example (for a given label index)
# ============================================================
def explain_example(
    predictor,
    shap_explainer,
    ig_explainer,
    row,
    example_dir,
    explanation_emotions,   # list of emotion indices
):
    example_dir.mkdir(parents=True, exist_ok=True)

    text = str(row[TEXT_COLUMN])

    # Save metadata
    # For multi‑label, we store all true and predicted labels
    true_labels = [label for label in LABELS if row[label] == 1]
    pred_labels = [label for label in LABELS if row[f"pred_{label}"] == 1]

    prediction_info = {
        "text": text,
        "true_labels": true_labels,
        "predicted_labels": pred_labels,
    }
    with (example_dir / "prediction.json").open("w", encoding="utf-8") as f:
        json.dump(prediction_info, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"TRUE LABELS: {true_labels}")
    print(f"PREDICTED LABELS: {pred_labels}")
    print(f"\nTEXT:\n{text}")

    # SHAP
    print("\nCalculating SHAP...")
    shap_values = explain_text(shap_explainer, text, max_evals=SHAP_MAX_EVALS)
    save_token_contributions(
        shap_values,
        str(example_dir / "shap_token_contributions.csv")
    )

    for emotion_idx in explanation_emotions:
        emotion = LABELS[emotion_idx]
        save_shap_html(shap_values, str(example_dir / f"shap_{emotion}_light.html"),
                       emotion_index=emotion_idx, dark_mode=False)
        save_shap_html(shap_values, str(example_dir / f"shap_{emotion}_dark.html"),
                       emotion_index=emotion_idx, dark_mode=True)

        top_tokens = get_top_tokens(shap_values, emotion, top_k=10)
        with (example_dir / f"shap_top_tokens_{emotion}.json").open("w", encoding="utf-8") as f:
            json.dump(top_tokens, f, indent=4, ensure_ascii=False)

    # Integrated Gradients
    for emotion_idx in explanation_emotions:
        emotion = LABELS[emotion_idx]
        print(f"Calculating IG for '{emotion}'...")
        ig_result = ig_explainer.explain(
            text=text,
            target_emotion_index=emotion_idx,
            n_steps=IG_STEPS,
        )
        save_ig_token_contributions(ig_result, str(example_dir / f"ig_{emotion}_tokens.csv"))
        save_ig_html(ig_result, str(example_dir / f"ig_{emotion}_light.html"), dark_mode=False)
        save_ig_html(ig_result, str(example_dir / f"ig_{emotion}_dark.html"), dark_mode=True)

    print(f"\nSaved explanations to: {example_dir}")