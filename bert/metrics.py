# project-root/bert/metrics.py
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
import numpy as np

def compute_all_metrics(preds_logits, trues, id2label, output_dict=False):
    """
    Computes classification report, confusion matrix, and macro scores.
    """
    preds_labels = np.argmax(preds_logits, axis=1)
    
    # Ensure target_names are strings
    target_names = [str(id2label[i]) for i in range(len(id2label))]

    report_str = classification_report(
        trues,
        preds_labels,
        target_names=target_names,
        digits=4,
        output_dict=False
    )
    report_dict = classification_report(
        trues,
        preds_labels,
        target_names=target_names,
        digits=4,
        output_dict=True
    )
    
    cm = confusion_matrix(trues, preds_labels)

    # Calculate macro scores separately to ensure they are always available
    f1_macro = f1_score(trues, preds_labels, average='macro', zero_division=0)
    precision_macro = precision_score(trues, preds_labels, average='macro', zero_division=0)
    recall_macro = recall_score(trues, preds_labels, average='macro', zero_division=0)

    return {
        "report": report_str,
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "f1_score_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }


def compute_risk_metrics(logits, labels):
    """
    Compute binary classification metrics for a single risk task.

    Args:
        logits: np.ndarray of shape (N,) or (N, 2)
        labels: np.ndarray of shape (N,) with values {0,1} (masked entries may be -100)

    Returns:
        dict with precision, recall, f1, auroc, support
    """

    # Mask ignored entries (-100)
    mask = labels != -100
    if mask.sum() == 0:
        # No valid labels
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
            "support": 0
        }

    labels = labels[mask]

    # Convert logits to probabilities
    if logits.ndim == 2:
        probs = logits[:, 1]
    else:
        probs = logits
    probs = probs[mask]

    preds = (probs >= 0.5).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0
    )

    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")  # Only one class present

    valid_classes = np.unique(labels)
    metric_valid = len(valid_classes) == 2 and len(labels) >= 30

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "support": int(np.sum(labels)),
        "valid": metric_valid,
        "class_balance": {
            "neg": int(np.sum(labels == 0)),
            "pos": int(np.sum(labels == 1)),
        }
    }


def threshold_sweep_metrics(labels, probs, thresholds=None):
    """
    Compute precision, recall, F1 for a range of thresholds for binary classification,
    safely ignoring masked labels (-100).
    """
    # Mask ignored entries
    mask = labels != -100
    labels = labels[mask]
    probs = probs[mask]

    if len(labels) == 0:
        # No valid labels, return empty metrics
        return {
            "thresholds": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

    thresholds = thresholds or np.linspace(0, 1, 101)
    metrics = {"thresholds": [], "precision": [], "recall": [], "f1": []}

    for t in thresholds:
        preds = (probs >= t).astype(int)
        metrics["thresholds"].append(t)
        metrics["precision"].append(precision_score(labels, preds, zero_division=0))
        metrics["recall"].append(recall_score(labels, preds, zero_division=0))
        metrics["f1"].append(f1_score(labels, preds, zero_division=0))

    return metrics
