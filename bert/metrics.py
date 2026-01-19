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
)
import numpy as np

def compute_all_metrics(preds, labels, id2label=None):
    """
    Computes accuracy, f1, confusion matrix, classification report, etc.
    id2label: dict mapping class indices to label names (optional)
    """
    preds = np.argmax(preds, axis=1)

    acc = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )

    # Use target_names if id2label is provided
    target_names = None
    if id2label is not None:
        target_names = [id2label[i] for i in range(len(id2label))]
    
    cls_report = classification_report(labels, preds, target_names=target_names, digits=4)
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": cls_report,
        "confusion_matrix": cm,
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
