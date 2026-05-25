# project-root/bert/visualize_metrics.py
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc


def plot_train_history(history, save_to=None, dark_mode=False):
    """
    Plot training and validation loss, and optionally emotion metrics over epochs.
    """
    if not history or not history.get("train_loss"):
        print("History is empty or contains no training loss. Skipping plot.")
        return

    bg_color = '#333333' if dark_mode else 'white'
    text_color = 'white' if dark_mode else 'black'

    pink_color = "#FEB2B4" if dark_mode else "#FF7F7F"
    yellow_color = "#FCD639" if dark_mode else "#F5D000"
    
    # Determine the number of subplots needed
    has_emotion_metrics = any(m is not None for m in history.get("emotion_metrics", []))
    num_subplots = 2 if has_emotion_metrics else 1
    
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 6 * num_subplots), facecolor=bg_color)
    if num_subplots == 1:
        axes = [axes] # Make it iterable

    # --- Plot 1: Loss ---
    ax1 = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "o-", color=pink_color, label="Training Loss")
    if history.get("val_loss") and not all(np.isnan(history["val_loss"])):
        ax1.plot(epochs, history["val_loss"], "o-", color=yellow_color, label="Validation Loss")
    
    ax1.set_title("Training and Validation Loss", color=text_color)
    ax1.set_xlabel("Epochs", color=text_color)
    ax1.set_ylabel("Loss", color=text_color)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_facecolor(bg_color)
    ax1.tick_params(axis='x', colors=text_color)
    ax1.tick_params(axis='y', colors=text_color)
    for spine in ax1.spines.values():
        spine.set_edgecolor(text_color)

    # --- Plot 2: Emotion Metrics (if available) ---

    mercury_color = "#BEC7B9" if dark_mode else "#819774"
    orange_color = "#F29668" if dark_mode else "#D16D3B"
    butteryellow_color = "#FFE497" if dark_mode else "#FFD769"

    if has_emotion_metrics:
        ax2 = axes[1]
        emotion_metrics = history.get("emotion_metrics", [])
        
        f1_scores = [m["f1_score_macro"] for m in emotion_metrics if m]
        precision_scores = [m["precision_macro"] for m in emotion_metrics if m]
        recall_scores = [m["recall_macro"] for m in emotion_metrics if m]

        if f1_scores:
            ax2.plot(epochs, f1_scores, "go-", color=mercury_color, label="Macro F1-Score")
        if precision_scores:
            ax2.plot(epochs, precision_scores, "yo-", color=orange_color, label="Macro Precision")
        if recall_scores:
            ax2.plot(epochs, recall_scores, "mo-", color=butteryellow_color, label="Macro Recall")

        ax2.set_title("Validation Emotion Metrics", color=text_color)
        ax2.set_xlabel("Epochs", color=text_color)
        ax2.set_ylabel("Score", color=text_color)
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.set_facecolor(bg_color)
        ax2.tick_params(axis='x', colors=text_color)
        ax2.tick_params(axis='y', colors=text_color)
        for spine in ax2.spines.values():
            spine.set_edgecolor(text_color)

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, facecolor=bg_color)
        print(f"Saved training history plot to {save_to}")
    else:
        plt.show()
    plt.close()

def plot_train_history_single(history, save_to=None, dark_mode=False):
    """
    history = {
        "train_loss": [...],
        "val_loss": [...],
        "val_f1": [...],
        "val_accuracy": [...]
    }
    """
    bg_color = '#333333' if dark_mode else 'white'
    text_color = 'white' if dark_mode else 'black'

    pink_color = "#FEB2B4" if dark_mode else "#FF7F7F"
    yellow_color = "#FCD639" if dark_mode else "#F5D000"

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg_color)

    ax.plot(history["train_loss"], label="Train Loss", color=pink_color)
    ax.plot(history["val_loss"], label="Validation Loss", color=yellow_color)
    
    # Update colors for val metrics based on dark mode 
    if "val_f1" in history:
        ax.plot(history["val_f1"], label="Val F1", linestyle="--")

    if "val_accuracy" in history:
        ax.plot(history["val_accuracy"], label="Val Accuracy", linestyle="--")

    ax.set_title("Training History", color=text_color)
    ax.set_xlabel("Epoch", color=text_color)
    ax.set_ylabel("Value", color=text_color)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor(bg_color)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, facecolor=bg_color)
        print(f"Saved single training history plot to {save_to}")
    else:
        plt.show()
    plt.close()


def save_classification_report(report: str, save_to: str):
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, "w", encoding="utf-8") as f:
        f.write(report)

def plot_confusion_matrix(cm, label_names, save_to=None, dark_mode=False, label_counts="absolute"):
    """
    Plot a confusion matrix using seaborn's heatmap.
    - Color gradient ALWAYS represents percentages (row-normalized)
    - Text shows either absolute counts or percentages based on label_counts
    """
    bg_color = '#333333' if dark_mode else 'white'
    text_color = 'white' if dark_mode else 'black'
    pink_color = "#FEB2B4" if dark_mode else "#FF7F7F"

    # Normalize for color coding (ALWAYS used for the heatmap values)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create custom annotations based on label_counts
    annotations = np.empty_like(cm, dtype='object')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if label_counts == "relative":
                annotations[i, j] = f"{cm_normalized[i, j]:.1%}"
            else:
                annotations[i, j] = str(int(cm[i, j]))

    # Set title and colorbar label
    if label_counts == "relative":
        title = "Confusion Matrix (%)"
    else:
        title = "Confusion Matrix"
    cbar_label = 'Percentage of Predictions (%)'

    cmap = LinearSegmentedColormap.from_list("custom_cmap", [bg_color, pink_color])

    plt.figure(figsize=(8, 6), facecolor=bg_color)
    sns.heatmap(
        cm_normalized,  # ALWAYS normalized for color!
        annot=annotations,  # Custom text annotations
        fmt='',  # Empty because we provide formatted annotations
        cmap=cmap,
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title(title, color=text_color)
    plt.ylabel("True Label", color=text_color)
    plt.xlabel("Predicted Label", color=text_color)
    plt.xticks(rotation=45, ha="center", color=text_color)
    plt.yticks(rotation=0, color=text_color)
    
    ax = plt.gca()
    ax.set_facecolor(bg_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(text_color)
    cbar.ax.tick_params(colors=text_color)
    cbar.set_label(cbar_label, color=bg_color)

    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, facecolor=bg_color, dpi=300)
        print(f"Saved confusion matrix to {save_to}")
    else:
        plt.show()
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
    plt.title(f"ROC Curve - {risk_name}")
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

