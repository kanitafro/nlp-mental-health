# project-root/bert/visualize_dashboard.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_combined_dashboard(history, risk_metrics_history, optimal_thresholds_history=None, save_to=None):
    """
    history: dict with 'train_loss', 'val_loss'
    risk_metrics_history: list of dicts per epoch, each dict: risk_name -> metric dict
    optimal_thresholds_history: optional list of dicts per epoch: risk_name -> threshold
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    risk_names = list(risk_metrics_history[0].keys()) if risk_metrics_history else []

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # ----------- Panel 1: Loss ----------
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    ax.plot(epochs, history["val_loss"], label="Val Loss", marker='o')
    ax.set_title("Emotion Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True) 
    
    # ----------- Panel 2: Emotion metrics (F1 example) ----------
    # For simplicity, if metrics history has per-epoch F1:
    if "train_metrics" in history and "val_metrics" in history:
        ax = axes[0, 1]
        ax.plot(epochs, [m["f1_macro"] for m in history["train_metrics"]], label="Train F1", marker='o')
        ax.plot(epochs, [m["f1_macro"] for m in history["val_metrics"]], label="Val F1", marker='o')
        ax.set_title("Emotion Classification F1 Macro")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.legend()
        ax.grid(True)
    else:
        axes[0, 1].axis('off')

    # ----------- Panel 3: Risk metrics ----------
    ax = axes[1, 0]
    for risk_name in risk_names:
        f1_scores = [epoch_rm[risk_name]['f1'] for epoch_rm in risk_metrics_history]
        auroc_scores = [epoch_rm[risk_name]['auroc'] for epoch_rm in risk_metrics_history]
        ax.plot(epochs, f1_scores, marker='o', label=f"{risk_name} F1")
        ax.plot(epochs, auroc_scores, marker='x', linestyle='--', label=f"{risk_name} AUROC")
    ax.set_title("Risk Metrics per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)

    # ----------- Panel 4: Optimal thresholds ----------
    ax = axes[1, 1]
    if optimal_thresholds_history:
        for risk_name in risk_names:
            thr_vals = []
            for epoch_data in optimal_thresholds_history:
                if risk_name in epoch_data:
                    thr_vals.append(epoch_data[risk_name]["thresholds"])
                else:
                    thr_vals.append(None)  # keeps epoch alignment

            ax.plot(epochs, thr_vals, marker='o', label=f"{risk_name} Threshold")
        ax.set_title("Optimal Thresholds per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Threshold")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)
    else:
        ax.axis('off')

    plt.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to)
        print(f"Dashboard saved to {save_to}")
    #plt.show()
