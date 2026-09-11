# project-root/bert/finetune_28.py

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bert.train import (
    get_device,
    EarlyStopping,
    create_optimizer,
    create_scheduler
)
from bert.dataset import MultiLabelGoEmotionsDataset
from bert.multilabel_model import MultiLabelBertModel
from bert.visualize_metrics import save_classification_report

# ---------------------------------------------------------------------
# Fixed GoEmotions 28-label order (same as in train.py)
# ---------------------------------------------------------------------
GOEMOTIONS_28_ORDER = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

# The 7 basic emotions in the same alphabetical order as Phase 1
BASIC_EMOTIONS_7 = ["anger", "disgust", "fear", "joy", "love", "sadness", "surprise"]

# Mapping from GoEmotions to one of the 7 basic emotions
# (neutral is intentionally left unmapped)
GOEMOTION_TO_BASIC = {
    "admiration": "love",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "love",
    "confusion": "surprise",
    "curiosity": "surprise",
    "desire": "love",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "fear",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "love",
    "grief": "sadness",
    "joy": "joy",
    "love": "love",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": None
}

# Pre-compute mapping matrix M (28 x 7) for auxiliary loss
def get_mapping_matrix(device):
    """Return the mapping matrix M (28 x 7) where M[i,j] = 1 if GoEmotion i maps to basic emotion j."""
    M = torch.zeros(len(GOEMOTIONS_28_ORDER), len(BASIC_EMOTIONS_7), device=device)
    for i, label in enumerate(GOEMOTIONS_28_ORDER):
        if label in GOEMOTION_TO_BASIC and GOEMOTION_TO_BASIC[label] is not None:
            basic = GOEMOTION_TO_BASIC[label]
            if basic in BASIC_EMOTIONS_7:
                j = BASIC_EMOTIONS_7.index(basic)
                M[i, j] = 1.0
    return M

def get_auxiliary_targets(labels_28, M):
    """
    Derive 7-label targets from 28-label targets.
    labels_28: (batch, 28) binary float
    M: (28, 7) mapping matrix
    Returns: (batch, 7) binary float
    """
    # For each basic emotion, it's 1 if ANY of its mapped GoEmotions is 1
    scores = labels_28 @ M  # (batch, 7)
    return (scores > 0.5).float()

# ---------------------------------------------------------------------
# Focal Loss (for multi-label)
# ---------------------------------------------------------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ---------------------------------------------------------------------
# Asymmetric Loss (CAL)
# ---------------------------------------------------------------------
class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, reduction='mean'):
        """
        Asymmetric Loss (CAL) for multi-label classification.
        - gamma_pos: typically 0 (no down-weighting for positives)
        - gamma_neg: typically 4 (down-weights easy negatives)
        - clip: prevents extreme gradients
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        # Positive loss (with gamma_pos, usually 0)
        loss_pos = -targets * torch.log(p + 1e-8) * (1 - p) ** self.gamma_pos
        # Negative loss (with gamma_neg to down-weight easy negatives)
        loss_neg = -(1 - targets) * torch.log(1 - p + 1e-8) * (1 - p) ** self.gamma_neg
        # Clip to prevent extreme gradients
        loss = torch.clamp(loss_pos + loss_neg, min=-self.clip, max=self.clip)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ---------------------------------------------------------------------
# Multi-label metrics (supports per-label thresholds)
# ---------------------------------------------------------------------
def compute_multilabel_metrics(labels_true, probs, thresholds=None):
    if thresholds is None:
        thresholds = 0.5
    if isinstance(thresholds, (int, float)):
        thresholds = np.full(probs.shape[1], thresholds)
    preds = (probs >= thresholds).astype(int)

    f1_micro = f1_score(labels_true, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels_true, preds, average="macro", zero_division=0)

    try:
        roc_auc_macro = roc_auc_score(
            labels_true, probs, average="macro", multi_class="ovr"
        )
    except ValueError:
        roc_auc_macro = float("nan")

    precision, recall, f1, support = precision_recall_fscore_support(
        labels_true, preds, average=None, zero_division=0
    )

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "roc_auc_macro": roc_auc_macro,
        "precision_per_label": precision,
        "recall_per_label": recall,
        "f1_per_label": f1,
        "support_per_label": support,
    }

# ---------------------------------------------------------------------
# Threshold tuning: find per-label threshold that maximizes F1
# ---------------------------------------------------------------------
def find_optimal_thresholds(labels_true, probs, n_thresholds=50, low=0.01, high=0.99):
    num_labels = probs.shape[1]
    optimal_thresholds = np.zeros(num_labels)
    for i in range(num_labels):
        y_true = labels_true[:, i]
        y_prob = probs[:, i]
        best_f1 = -1.0
        best_thr = 0.5
        for thr in np.linspace(low, high, n_thresholds):
            y_pred = (y_prob >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        optimal_thresholds[i] = best_thr
    return optimal_thresholds

# ---------------------------------------------------------------------
# Generate multi-label classification report (text)
# ---------------------------------------------------------------------
def generate_multilabel_report(metrics, label_names):
    lines = []
    lines.append("              precision    recall  f1-score   support")
    lines.append("")
    for i, label in enumerate(label_names):
        p = metrics["precision_per_label"][i]
        r = metrics["recall_per_label"][i]
        f = metrics["f1_per_label"][i]
        s = metrics["support_per_label"][i]
        lines.append(f"{label:>15}  {p:8.4f}  {r:7.4f}  {f:8.4f}  {s:8d}")
    lines.append("")
    lines.append(f"{'micro avg':>15}  {metrics['f1_micro']:8.4f}  {metrics['f1_micro']:8.4f}  {metrics['f1_micro']:8.4f}  -")
    lines.append(f"{'macro avg':>15}  {metrics['f1_macro']:8.4f}  {metrics['f1_macro']:8.4f}  {metrics['f1_macro']:8.4f}  -")
    return "\n".join(lines)

# ---------------------------------------------------------------------
# Plot per-label metrics (light & dark)
# ---------------------------------------------------------------------
def plot_per_label_metrics(metrics, label_names, save_to, dark_mode=False):
    pink_light = "#FF7F7F"
    yellow_light = "#F5D000"
    mercury_light = "#819774"
    pink_dark = "#FEB2B4"
    yellow_dark = "#FCD639"
    mercury_dark = "#BEC7B9"

    if dark_mode:
        bg_color = '#333333'
        text_color = 'white'
        pink = pink_dark
        yellow = yellow_dark
        mercury = mercury_dark
    else:
        bg_color = 'white'
        text_color = 'black'
        pink = pink_light
        yellow = yellow_light
        mercury = mercury_light

    x = np.arange(len(label_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 8), facecolor=bg_color)
    ax.bar(x - width, metrics["precision_per_label"], width, label='Precision', color=pink)
    ax.bar(x, metrics["recall_per_label"], width, label='Recall', color=yellow)
    ax.bar(x + width, metrics["f1_per_label"], width, label='F1', color=mercury)

    ax.set_xlabel('Emotion Label', color=text_color)
    ax.set_ylabel('Score', color=text_color)
    ax.set_title('Per‑Label Performance on Test Set', color=text_color)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=90, color=text_color)
    ax.tick_params(axis='y', colors=text_color)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_facecolor(bg_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(text_color)

    plt.tight_layout()
    plt.savefig(save_to, facecolor=bg_color, dpi=300)
    plt.close()

# ---------------------------------------------------------------------
# Plot training history (loss & F1) - light & dark
# ---------------------------------------------------------------------
def plot_training_history(history, save_to, dark_mode=False):
    pink_light = "#FF7F7F"
    yellow_light = "#F5D000"
    mercury_light = "#819774"
    pink_dark = "#FEB2B4"
    yellow_dark = "#FCD639"
    mercury_dark = "#BEC7B9"
    orange_light = "#D16D3B"
    orange_dark = "#F29668"

    if dark_mode:
        bg_color = '#333333'
        text_color = 'white'
        pink = pink_dark
        yellow = yellow_dark
        mercury = mercury_dark
        orange = orange_dark
    else:
        bg_color = 'white'
        text_color = 'black'
        pink = pink_light
        yellow = yellow_light
        mercury = mercury_light
        orange = orange_light

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=bg_color)

    ax1.plot(epochs, history["train_loss"], 'o-', label="Train Loss", color=pink)
    ax1.set_xlabel("Epoch", color=text_color)
    ax1.set_ylabel("Loss", color=text_color)
    ax1.set_title("Training Loss", color=text_color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_facecolor(bg_color)
    ax1.tick_params(colors=text_color)
    for spine in ax1.spines.values():
        spine.set_edgecolor(text_color)

    ax2.plot(epochs, history["val_f1_macro"], 'o-', label="Macro F1", color=yellow)
    ax2.plot(epochs, history["val_f1_micro"], 's-', label="Micro F1", color=mercury)
    ax2.set_xlabel("Epoch", color=text_color)
    ax2.set_ylabel("F1 Score", color=text_color)
    ax2.set_title("Validation F1", color=text_color)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_facecolor(bg_color)
    ax2.tick_params(colors=text_color)
    for spine in ax2.spines.values():
        spine.set_edgecolor(text_color)

    plt.tight_layout()
    plt.savefig(save_to, facecolor=bg_color, dpi=300)
    plt.close()

# ---------------------------------------------------------------------
# Training / Validation functions
# ---------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn, M, aux_weight):
    model.train()
    total_loss = 0.0
    aux_loss_fn = torch.nn.BCEWithLogitsLoss()  # auxiliary loss uses BCE

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_28 = batch["labels"].to(device)  # (batch, 28)

        optimizer.zero_grad()

        logits, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )

        # Main loss (28-class)
        main_loss = loss_fn(logits, labels_28)

        # Auxiliary loss (7-class)
        if M is not None and aux_weight > 0:
            # Derive 7-label targets from 28-label targets
            labels_7 = get_auxiliary_targets(labels_28, M)
            # Project 28 logits to 7 logits via mapping matrix M
            logits_7 = logits @ M  # (batch, 7)
            aux_loss = aux_loss_fn(logits_7, labels_7)
            loss = main_loss + aux_weight * aux_loss
        else:
            loss = main_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
            )

            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_probs, all_labels

# ---------------------------------------------------------------------
# Function to freeze/unfreeze backbone
# ---------------------------------------------------------------------
def set_backbone_trainable(model, trainable):
    for param in model.bert.parameters():
        param.requires_grad = trainable
    print(f"Backbone trainable: {trainable}")

# ---------------------------------------------------------------------
# Hierarchical initialisation
# ---------------------------------------------------------------------
def apply_hierarchical_initialisation(model, label_columns, checkpoint_path, device):
    """
    Load the 7‑emotion checkpoint, extract the old classifier weights,
    and initialise the 28‑class classifier using the mapping defined above.
    """
    # Load checkpoint (on CPU to avoid memory issues)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    old_weight = checkpoint.get('emotion_classifier.weight', None)
    old_bias = checkpoint.get('emotion_classifier.bias', None)

    if old_weight is None:
        print("WARNING: 'emotion_classifier.weight' not found in checkpoint. Skipping hierarchical initialisation.")
        return

    # Check that old_weight has shape (7, hidden)
    if old_weight.shape[0] != len(BASIC_EMOTIONS_7):
        print(f"WARNING: Expected 7 basic emotions, but checkpoint has {old_weight.shape[0]} classes. Skipping.")
        return

    print(f"Initialising 28‑class classifier using hierarchical mapping from {len(BASIC_EMOTIONS_7)} basic emotions.")

    # Create mapping matrix M (28 x 7)
    num_labels = len(label_columns)
    M = torch.zeros(num_labels, len(BASIC_EMOTIONS_7), device=device)

    for i, label in enumerate(label_columns):
        if label in GOEMOTION_TO_BASIC and GOEMOTION_TO_BASIC[label] is not None:
            basic = GOEMOTION_TO_BASIC[label]
            if basic in BASIC_EMOTIONS_7:
                j = BASIC_EMOTIONS_7.index(basic)
                M[i, j] = 1.0

    # Move old weights to device
    old_weight = old_weight.to(device)
    if old_bias is not None:
        old_bias = old_bias.to(device)

    with torch.no_grad():
        # Compute new weights: W_28 = M @ W_7
        new_weight = M @ old_weight  # (28, hidden)
        # For rows that are all zero (neutral), we keep the original random initialisation.
        # We'll only overwrite rows that have a mapping.
        for i, label in enumerate(label_columns):
            if label in GOEMOTION_TO_BASIC and GOEMOTION_TO_BASIC[label] is not None:
                basic = GOEMOTION_TO_BASIC[label]
                if basic in BASIC_EMOTIONS_7:
                    model.classifier.weight[i, :] = new_weight[i, :]
                    if old_bias is not None:
                        basic_idx = BASIC_EMOTIONS_7.index(basic)
                        model.classifier.bias[i] = old_bias[basic_idx]
                    else:
                        model.classifier.bias[i] = 0.0

    print("Hierarchical initialisation applied successfully.")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune 7-emotion DistilBERT on GoEmotions (28 multi-label)."
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/processed/goemotions/goemotions_merged_preprocessed.csv",
        help="Path to merged GoEmotions CSV with 'split' column.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="clean_text_transf",
        help="Column containing preprocessed text.",
    )
    parser.add_argument(
        "--checkpoint_7",
        type=str,
        default="checkpoints_v2_7/best_model.pt",
        help="Path to the trained 7-emotion model (best_model.pt).",
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Base transformer model name.",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)

    # Loss function options
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use Focal Loss instead of BCE with pos_weight."
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss (default 2.0)."
    )
    parser.add_argument(
        "--focal_alpha",
        action="store_true",
        help="Alpha parameter for Focal Loss (per-label weight). If None, uses pos_weight."
    )

    # Asymmetric Loss (CAL)
    parser.add_argument(
        "--use_asymmetric_loss",
        action="store_true",
        help="Use Asymmetric Loss (CAL) instead of BCE with pos_weight."
    )
    parser.add_argument(
        "--asymmetric_gamma_neg",
        type=float,
        default=4.0,
        help="Gamma negative for Asymmetric Loss (default 4.0)."
    )
    parser.add_argument(
        "--asymmetric_clip",
        type=float,
        default=0.05,
        help="Clipping value for Asymmetric Loss (default 0.05)."
    )

    # Auxiliary loss (Theory 2)
    parser.add_argument(
        "--use_auxiliary_loss",
        action="store_true",
        help="Add auxiliary 7-class loss using the hierarchical mapping."
    )
    parser.add_argument(
        "--auxiliary_loss_weight",
        type=float,
        default=0.3,
        help="Weight for the auxiliary loss (default 0.3)."
    )

    # Threshold tuning
    parser.add_argument(
        "--tune_thresholds",
        action="store_true",
        default=True,
        help="Tune per-label thresholds on validation set (default True)."
    )
    parser.add_argument(
        "--threshold_search_steps",
        type=int,
        default=50,
        help="Number of threshold steps for tuning (default 50)."
    )
    parser.add_argument(
        "--threshold_search_low",
        type=float,
        default=0.01,
        help="Lower bound for threshold search (default 0.01)."
    )
    parser.add_argument(
        "--threshold_search_high",
        type=float,
        default=0.99,
        help="Upper bound for threshold search (default 0.99)."
    )

    # Gradual unfreezing
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Number of initial epochs to keep the backbone frozen (classifier head only trains). After that, unfreeze all."
    )

    # Hierarchical initialisation
    parser.add_argument(
        "--use_hierarchical_init",
        action="store_true",
        help="Initialise the 28‑class classifier using the 7‑emotion classifier weights via a semantic mapping."
    )

    # Split
    parser.add_argument(
        "--use_split_column",
        action="store_true",
        default=True,
        help="Use the 'split' column from the CSV (train/dev/test).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="If not using split column, fraction for validation.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="If not using split column, fraction for test.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model and results. If None, uses model_version.",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="goemotions_v1",
        help="Version string used to create output directory if output_dir is not set.",
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=None,
        help="Learning rate for the backbone (if different from head). Default: use --learning_rate."
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        default=False,
        help="Whether to oversample minority classes in the training dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"saved_models/finetuned_model_{args.model_version}"
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    # Pre-compute mapping matrix M for auxiliary loss
    M = get_mapping_matrix(device) if args.use_auxiliary_loss else None

    # -----------------------------------------------------------------
    # 1. Load Data
    # -----------------------------------------------------------------
    df = pd.read_csv(args.data_path)

    label_columns = [col for col in GOEMOTIONS_28_ORDER if col in df.columns]
    missing = set(GOEMOTIONS_28_ORDER) - set(label_columns)
    if missing:
        raise ValueError(f"Missing label columns in CSV: {missing}")
    print(f"Found {len(label_columns)} label columns.")

    if args.use_split_column and "split" in df.columns:
        print("Using 'split' column for train/dev/test.")
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "dev"].copy()
        test_df = df[df["split"] == "test"].copy()
        print(f"Train: {len(train_df)}, Dev: {len(val_df)}, Test: {len(test_df)}")
    else:
        print("Using random split.")
        from sklearn.model_selection import train_test_split
        train_val, test_df = train_test_split(
            df, test_size=args.test_size, random_state=args.seed
        )
        val_size_adjusted = args.val_size / (1 - args.test_size)
        train_df, val_df = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=args.seed
        )
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # -----------------------------------------------------------------
    # 2. Tokenizer
    # -----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_length

    # -----------------------------------------------------------------
    # 3. Datasets & DataLoaders
    # -----------------------------------------------------------------
    train_ds = MultiLabelGoEmotionsDataset(
        train_df, tokenizer, args.max_length, label_columns, args.text_column, oversample=args.oversample
    )
    val_ds = MultiLabelGoEmotionsDataset(
        val_df, tokenizer, args.max_length, label_columns, args.text_column, oversample=False
    )
    test_ds = MultiLabelGoEmotionsDataset(
        test_df, tokenizer, args.max_length, label_columns, args.text_column, oversample=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # -----------------------------------------------------------------
    # 4. Define loss function
    # -----------------------------------------------------------------
    if args.use_focal_loss:
        if not args.focal_alpha:
            loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=None, reduction='mean')
            print(f"Using Focal Loss with gamma={args.focal_gamma} (no alpha/pos_weight).")
        else:
            pos_counts = train_df[label_columns].sum(axis=0).values
            neg_counts = len(train_df) - pos_counts
            alpha = neg_counts / pos_counts
            alpha = np.nan_to_num(alpha, nan=1.0, posinf=1.0, neginf=1.0)
            alpha = torch.tensor(alpha, dtype=torch.float, device=device)
            loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=alpha, reduction='mean')
            print(f"Using Focal Loss with gamma={args.focal_gamma} and alpha (pos_weight) per label.")
    elif args.use_asymmetric_loss:
        loss_fn = AsymmetricLoss(
            gamma_pos=0,
            gamma_neg=args.asymmetric_gamma_neg,
            clip=args.asymmetric_clip,
            reduction='mean'
        )
        print(f"Using Asymmetric Loss with gamma_neg={args.asymmetric_gamma_neg}, clip={args.asymmetric_clip}")
    else:
        pos_counts = train_df[label_columns].sum(axis=0).values
        neg_counts = len(train_df) - pos_counts
        pos_weight = torch.tensor(
            neg_counts / pos_counts, dtype=torch.float, device=device
        )
        pos_weight = torch.nan_to_num(pos_weight, nan=1.0, posinf=1.0, neginf=1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Using BCEWithLogitsLoss with pos_weight per label.")

    # -----------------------------------------------------------------
    # 5. Model
    # -----------------------------------------------------------------
    if not os.path.exists(args.checkpoint_7):
        print(f"WARNING: Checkpoint {args.checkpoint_7} not found. Starting from scratch.")
        model = MultiLabelBertModel.from_pretrained(
            args.model_name,
            num_labels=len(label_columns),
            dropout_rate=args.dropout_rate,
            ignore_mismatched_sizes=True,
        ).to(device)
    else:
        print(f"Loading 7‑emotion checkpoint from {args.checkpoint_7} ...")
        model = MultiLabelBertModel.load_7emotion_weights(
            checkpoint_path=args.checkpoint_7,
            model_name=args.model_name,
            num_labels=len(label_columns),
            dropout_rate=args.dropout_rate,
            device=device,
        )

    # Apply hierarchical initialisation if requested
    if args.use_hierarchical_init and os.path.exists(args.checkpoint_7):
        apply_hierarchical_initialisation(
            model, label_columns, args.checkpoint_7, device
        )

    # Initially, set backbone trainable to True (will be frozen later if needed)
    set_backbone_trainable(model, trainable=True)

    # -----------------------------------------------------------------
    # 6. Optimizer (with separate learning rates)
    # -----------------------------------------------------------------
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "bert" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    head_lr = args.learning_rate
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else head_lr

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": args.weight_decay}
    ])

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"Model: {args.model_name} with dropout {args.dropout_rate}")
    print(f"    Training for {args.epochs} epochs with batch size {args.batch_size}")
    print(f"    Loss function: {'Focal Loss' if args.use_focal_loss else 'Asymmetric Loss' if args.use_asymmetric_loss else 'BCEWithLogitsLoss'}")
    print(f"    Learning rates: head_lr={head_lr}, backbone_lr={backbone_lr}")
    print(f"Oversampling: {args.oversample}")
    print(f"Hierarchical init: {args.use_hierarchical_init}")
    print(f"Auxiliary loss: {args.use_auxiliary_loss} (weight={args.auxiliary_loss_weight})")

    # -----------------------------------------------------------------
    # 7. Training Loop
    # -----------------------------------------------------------------
    best_val_f1 = -1.0
    early_stopper = EarlyStopping(patience=args.patience)

    history = {"train_loss": [], "val_f1_micro": [], "val_f1_macro": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # Gradual unfreezing
        if args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs:
            set_backbone_trainable(model, trainable=False)
        else:
            if epoch == args.freeze_backbone_epochs + 1:
                set_backbone_trainable(model, trainable=True)

        # Train one epoch (with auxiliary loss if enabled)
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn,
            M, args.auxiliary_loss_weight if args.use_auxiliary_loss else 0.0
        )

        # Validation
        val_probs, val_labels = validate_epoch(model, val_loader, device)

        val_metrics_05 = compute_multilabel_metrics(val_labels, val_probs, thresholds=0.5)
        val_f1_micro = val_metrics_05["f1_micro"]
        val_f1_macro = val_metrics_05["f1_macro"]

        history["train_loss"].append(train_loss)
        history["val_f1_micro"].append(val_f1_micro)
        history["val_f1_macro"].append(val_f1_macro)

        print(
            f"Train Loss: {train_loss:.4f} | Val F1(micro): {val_f1_micro:.4f} | Val F1(macro): {val_f1_macro:.4f}"
        )
        try:
            roc_auc = roc_auc_score(val_labels, val_probs, average="macro", multi_class="ovr")
            print(f"Val ROC-AUC(macro): {roc_auc:.4f}")
        except ValueError:
            pass

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"✅ New best model saved (macro F1 = {best_val_f1:.4f})")

        early_stopper.step(-val_f1_macro)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # -----------------------------------------------------------------
    # 8. Load Best Model & Threshold Tuning
    # -----------------------------------------------------------------
    best_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_path):
        print(f"\nLoading best model from {best_path} for evaluation.")
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print("\nNo best model found, using final model.")

    val_probs, val_labels = validate_epoch(model, val_loader, device)

    if args.tune_thresholds:
        print("\nTuning per-label thresholds on validation set...")
        optimal_thresholds = find_optimal_thresholds(
            val_labels, val_probs,
            n_thresholds=args.threshold_search_steps,
            low=args.threshold_search_low,
            high=args.threshold_search_high
        )
        np.save(os.path.join(args.output_dir, "optimal_thresholds.npy"), optimal_thresholds)
        print("Optimal thresholds:")
        for i in range(len(label_columns)):
            print(f"  {label_columns[i]}: {optimal_thresholds[i]:.3f}")
    else:
        optimal_thresholds = None

    print("\nEvaluating on test set...")
    test_probs, test_labels = validate_epoch(model, test_loader, device)

    if optimal_thresholds is not None:
        test_metrics = compute_multilabel_metrics(test_labels, test_probs, thresholds=optimal_thresholds)
        print("\n=== Test Results (with tuned thresholds) ===")
    else:
        test_metrics = compute_multilabel_metrics(test_labels, test_probs, thresholds=0.5)
        print("\n=== Test Results (with default 0.5 threshold) ===")

    print(f"F1 (micro): {test_metrics['f1_micro']:.4f}")
    print(f"F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"ROC-AUC (macro): {test_metrics['roc_auc_macro']:.4f}")

    print("\nPer-label F1:")
    for idx, label in enumerate(label_columns):
        print(f"  {label}: {test_metrics['f1_per_label'][idx]:.4f}")

    # -----------------------------------------------------------------
    # 9. Save All Outputs
    # -----------------------------------------------------------------
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        metrics_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in test_metrics.items()
        }
        json.dump(metrics_serializable, f, indent=4)

    report_str = generate_multilabel_report(test_metrics, label_columns)
    save_classification_report(report_str, os.path.join(args.output_dir, "classification_report.txt"))

    plot_per_label_metrics(
        test_metrics,
        label_columns,
        os.path.join(args.output_dir, "per_label_metrics_light.png"),
        dark_mode=False
    )
    plot_per_label_metrics(
        test_metrics,
        label_columns,
        os.path.join(args.output_dir, "per_label_metrics_dark.png"),
        dark_mode=True
    )

    plot_training_history(
        history,
        os.path.join(args.output_dir, "training_history_light.png"),
        dark_mode=False
    )
    plot_training_history(
        history,
        os.path.join(args.output_dir, "training_history_dark.png"),
        dark_mode=True
    )

    np.savez(
        os.path.join(args.output_dir, "test_predictions.npz"),
        probs=test_probs,
        labels=test_labels,
        thresholds=optimal_thresholds if optimal_thresholds is not None else np.full(len(label_columns), 0.5)
    )

    print(f"\nAll results saved to {args.output_dir}")
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()