# project-root/bert/visualize_metrics.py
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(cm, labels=None, label_names=None, save_to=None):
    if label_names is None:
        label_names = labels  # fallback to integer labels

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, bbox_inches="tight")
    plt.close()


def plot_train_history(history, save_to=None):
    """
    history = {
        "train_loss": [...],
        "val_loss": [...],
        "val_f1": [...],
        "val_accuracy": [...]
    }
    """
    plt.figure(figsize=(8, 6))

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    if "val_f1" in history:
        plt.plot(history["val_f1"], label="Val F1", linestyle="--")

    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="Val Accuracy", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.legend()

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, bbox_inches="tight")
    plt.close()


def save_classification_report(report: str, save_to: str):
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, "w", encoding="utf-8") as f:
        f.write(report)


def plot_risk_metrics(risk_metrics: dict, save_dir: str):
    """
    risk_metrics = {
        "depression": {"precision": ..., "recall": ..., "f1": ..., "auroc": ...},
        "selfharm": {...},
        ...
    }
    """

    os.makedirs(save_dir, exist_ok=True)

    for risk_name, metrics in risk_metrics.items():
        values = [
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
            metrics.get("f1", 0.0),
            metrics.get("auroc", 0.0),
        ]

        labels = ["Precision", "Recall", "F1", "AUROC"]

        plt.figure(figsize=(6, 4))
        sns.barplot(x=labels, y=values)
        plt.ylim(0, 1)
        plt.title(f"Risk Metrics – {risk_name}")

        save_path = os.path.join(save_dir, f"risk_metrics_{risk_name}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

def plot_risk_roc_curve(labels, probs, risk_name, save_to=None):
    """
    Plot ROC curve for a single risk task.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {risk_name}")
    plt.legend(loc="lower right")

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, bbox_inches="tight")

    plt.close()

def plot_combined_risk_roc_curves(risk_outputs, save_to=None):
    """
    Plot ROC curves for all risks in a single figure.
    """
    plt.figure(figsize=(8, 6))

    for risk_name, data in risk_outputs.items():
        labels = data["labels"]
        probs = data["probs"]

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{risk_name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – Risk Tasks")
    plt.legend(loc="lower right")

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, bbox_inches="tight")

    plt.close()

def plot_threshold_sweep(metrics_sweep, risk_name, save_to=None):
    """
    Plot precision, recall, F1 vs thresholds and mark optimal threshold (max F1).
    Returns the optimal threshold (max F1).
    """
    thresholds = metrics_sweep["thresholds"]
    precision = metrics_sweep["precision"]
    recall = metrics_sweep["recall"]
    f1 = metrics_sweep["f1"]

    # Find optimal threshold
    optimal_idx = np.argmax(f1)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1[optimal_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision, label="Precision", marker="o")
    plt.plot(thresholds, recall, label="Recall", marker="o")
    plt.plot(thresholds, f1, label="F1", marker="o")
    
    # Mark optimal F1
    plt.scatter(optimal_threshold, optimal_f1, color="red", s=100, zorder=5)
    plt.text(optimal_threshold, optimal_f1 + 0.02,
             f"Optimal: {optimal_threshold:.2f}", color="red", ha="center")

    plt.title(f"Threshold Sweep - {risk_name}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    if save_to:
        plt.savefig(save_to, bbox_inches="tight")
    plt.close()
    
    return optimal_threshold

