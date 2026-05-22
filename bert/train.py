# project-root/bert/train.py
import argparse
import pandas as pd
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
# python train.py --labels 7 --debug --emotion_cv_folds 5

from visualize_dashboard import plot_combined_dashboard
from dataset import TextDataset
from model_utils import create_optimizer, create_scheduler, EarlyStopping
from metrics import compute_all_metrics, compute_risk_metrics, threshold_sweep_metrics
from visualize_metrics import plot_combined_risk_roc_curves, plot_confusion_matrix, plot_risk_roc_curve, plot_threshold_sweep, plot_train_history, save_classification_report, plot_risk_metrics
from lexicon_utils import ThemeLexicon, SubthemeInferencer
from multitask_model import BertEmotionRiskModel
from inference import predict_chunked

DEFAULT_6_LABEL_PATH = "../data/processed/dataset_6labels_clean_more.csv"
DEFAULT_7_LABEL_PATH = "../data/processed/dataset_7labels_clean.csv"
DEFAULT_28_LABEL_PATH = "../data/processed/goemotions.csv" # to add later
DEFAULT_LEXICON_PATH_6 = "../data/lexicon/lexicon_clean_6.json"
DEFAULT_LEXICON_PATH_7 = "../data/lexicon/lexicon_clean_7.json"
DEFAULT_LEXICON_PATH_28 = "../data/lexicon/lexicon_clean.json"

DEFAULT_EMOTION_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_RISK_MODEL_NAME = "prajjwal1/bert-mini"

# =========================
# Risk flag dataset paths
# =========================

DEFAULT_RISK_DATASETS = {
    "depression": "../data/processed/dataset_depression_clean.csv",
    "selfharm": "../data/processed/dataset_selfharm_clean.csv",
    "suicidal": "../data/processed/dataset_suicidal_clean.csv",
    "grief": "../data/processed/dataset_grief_clean.csv",
}


GOEMOTIONS_28_ORDER = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("\nCUDA not available, using CPU")
        return torch.device("cpu")
    
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory - Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"GPU Memory - Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


# =========================
# Training helpers
# =========================
def init_risk_iterators(risk_loaders):
    """
    Creates persistent iterators for each risk dataset.
    """
    return {
        name: iter(loader)
        for name, loader in risk_loaders
    }

def train_batch(model, batch, criterion, device, accumulation_steps,
                use_emotions=True, risk_iters=None, risk_loaders=None, risk_loss_weight=1.0):

    batch = {k: v.to(device) for k, v in batch.items()}

    total_loss = None

    # ---- Emotion loss (optional) ----
    if use_emotions:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=None,
            risk_labels=None
        )
        emotion_labels = batch["emotion_labels"]
        emotion_loss = criterion(outputs.logits, emotion_labels) if criterion is not None else outputs.loss
        total_loss = emotion_loss

    # ---- Risk multitask loss (one batch per dataset) ----
    if risk_iters:
        for risk_name, riter in risk_iters.items():
            try:
                rbatch = next(riter)
            except StopIteration:
                # Re-create a fresh iterator from the original DataLoader
                riter = iter(risk_loaders[risk_name]["train"])
                risk_iters[risk_name] = riter
                rbatch = next(riter)

            rbatch = {k: v.to(device) for k, v in rbatch.items()}

            risk_out = model(
                input_ids=rbatch["input_ids"],
                attention_mask=rbatch["attention_mask"],
                labels=None,
                risk_labels=rbatch["risk_labels"]
            )

            idx = model.risk_names.index(risk_name)
            risk_logits = risk_out.risk_logits[:, idx]
            risk_labels = rbatch["risk_labels"][:, idx].float()

            risk_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                risk_logits,
                risk_labels
            )

            weighted_risk_loss = risk_loss_weight * risk_loss
            total_loss = weighted_risk_loss if total_loss is None else total_loss + weighted_risk_loss

    if total_loss is None:
        raise ValueError("No active training objective. Enable --use_emotions and/or --use_risk_flags.")

    total_loss = total_loss / accumulation_steps
    total_loss.backward()

    return total_loss.item()


def validate_emotions_only(model, val_loader, device, args):
    """
    Validate only the main emotion dataset.
    Returns avg_val_loss and emotion metrics.
    """
    model.eval()
    val_loss = 0.0
    preds, trues = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation (Emotions)", leave=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["emotion_labels"],
                risk_labels=None
            )

            val_loss += outputs.loss.item()
            preds.append(outputs.logits.cpu().numpy())
            trues.append(batch["emotion_labels"].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    metrics = compute_all_metrics(
        preds,
        trues,
        id2label=args._id2label
    )

    return avg_val_loss, metrics


def validate_risk_datasets(model, risk_loaders, device, split="val", debug=False):
    """
    Forward pass each risk dataset separately.
    Returns risk_metrics, risk_outputs
    """
    risk_metrics = {}
    risk_outputs = {}

    if not risk_loaders:
        return risk_metrics, risk_outputs

    model.eval()
    for risk_name, loaders in risk_loaders.items():
        r_loader = loaders[split]
        all_logits, all_labels = [], []

        # Optional debug subset
        if debug:
            r_loader = DataLoader(
                list(r_loader.dataset)[:50],
                batch_size=r_loader.batch_size,
                shuffle=False
            )

        with torch.no_grad():
            for rbatch in tqdm(r_loader, desc=f"Risk {split.title()} ({risk_name})", leave=True):
                rbatch = {k: v.to(device) for k, v in rbatch.items()}
                outputs = model(
                    input_ids=rbatch["input_ids"],
                    attention_mask=rbatch["attention_mask"],
                    labels=None,
                    risk_labels=rbatch["risk_labels"]
                )
                idx = model.risk_names.index(risk_name)
                all_logits.append(outputs.risk_logits[:, idx].cpu().numpy())
                all_labels.append(rbatch["risk_labels"][:, idx].cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        rm = compute_risk_metrics(all_logits, all_labels)

        n_samples = len(all_labels)
        allow_roc = n_samples >= 100 and len(np.unique(all_labels)) == 2

        if not allow_roc:
            rm["auroc"] = float("nan")
            rm["valid"] = False
            rm["note"] = "Skipped ROC (debug / insufficient data)"
        else:
            rm["valid"] = True

        risk_metrics[risk_name] = rm

        risk_outputs[risk_name] = {"labels": all_labels, "probs": expit(all_logits)}

    return risk_metrics, risk_outputs


def build_risk_cv_folds(rdf, n_splits, test_size, random_state=42):
    """
    Split a risk dataframe into a held-out test set and stratified CV folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation")

    if test_size > 0:
        trainval_df, test_df = train_test_split(
            rdf,
            test_size=test_size,
            stratify=rdf["label"],
            random_state=random_state,
        )
    else:
        trainval_df = rdf.copy()
        test_df = None

    label_counts = trainval_df["label"].value_counts()
    if label_counts.min() < n_splits:
        raise ValueError(
            f"Cannot make {n_splits}-fold CV for this risk dataset because the smallest class has only {label_counts.min()} samples after the held-out test split."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for train_idx, val_idx in skf.split(trainval_df, trainval_df["label"]):
        folds.append(
            {
                "train_df": trainval_df.iloc[train_idx].copy(),
                "val_df": trainval_df.iloc[val_idx].copy(),
            }
        )

    return trainval_df, test_df, folds


def summarize_metric_dicts(metric_dicts):
    """
    Summarize a list of metric dictionaries into mean/std per numeric key.
    """
    summary = {}
    if not metric_dicts:
        return summary

    keys = set()
    for metric_dict in metric_dicts:
        keys.update(metric_dict.keys())

    for key in keys:
        values = []
        for metric_dict in metric_dicts:
            value = metric_dict.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and not np.isnan(value):
                values.append(float(value))
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    return summary


def run_emotion_cross_validation(df, create_dataset_fn, selected_model_name, args, device):
    """
    Run stratified CV for emotion-only training.
    """
    cv_args = argparse.Namespace(**vars(args))
    cv_args.save_checkpoints = False
    cv_args.early_stopping = False

    skf = StratifiedKFold(n_splits=cv_args.emotion_cv_folds, shuffle=True, random_state=42)
    
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"\n=== Emotion CV Fold {fold_idx + 1}/{cv_args.emotion_cv_folds} ===")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_ds = create_dataset_fn(train_df)
        val_ds = create_dataset_fn(val_df)

        train_loader = DataLoader(train_ds, batch_size=cv_args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cv_args.batch_size, shuffle=False)

        model = BertEmotionRiskModel(
            selected_model_name,
            num_labels=len(args._label2id),
            use_risk=False,  # Emotion CV is emotion-only
            dropout_rate=args.dropout_rate,
        ).to(device)

        total_steps = len(train_loader) * cv_args.epochs
        warmup_steps = int(cv_args.warmup_ratio * total_steps)
        optimizer = create_optimizer(model, lr=cv_args.learning_rate, weight_decay=0.05)
        scheduler = create_scheduler(optimizer, warmup_steps, total_steps)

        val_metrics, _, _, _ = train_one_fold(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            cv_args,
            epoch_start=0,
            train_df=train_df,
            risk_loaders=None, # No risk loaders in emotion CV
        )
        
        fold_results.append(val_metrics)

        print(f"\nFold {fold_idx + 1} Validation Metrics:")
        print(val_metrics["report"])


    # Summarize results
    all_reports = [res["report_dict"] for res in fold_results]
    
    summary = {}
    for label in all_reports[0].keys():
        if isinstance(all_reports[0][label], dict):
            summary[label] = {
                "precision": np.mean([r[label]["precision"] for r in all_reports]),
                "recall": np.mean([r[label]["recall"] for r in all_reports]),
                "f1-score": np.mean([r[label]["f1-score"] for r in all_reports]),
                "support": np.mean([r[label]["support"] for r in all_reports]),
            }

    # Macro and weighted averages
    for avg_type in ["macro avg", "weighted avg"]:
        summary[avg_type] = {
            "precision": np.mean([r[avg_type]["precision"] for r in all_reports]),
            "recall": np.mean([r[avg_type]["recall"] for r in all_reports]),
            "f1-score": np.mean([r[avg_type]["f1-score"] for r in all_reports]),
        }

    summary["accuracy"] = np.mean([r["accuracy"] for r in all_reports])

    print("\n=== Emotion Cross-Validation Summary ===")
    print(f"Ran {cv_args.emotion_cv_folds} folds.")
    
    # Pretty print the summary
    for label, metrics in summary.items():
        if isinstance(metrics, dict):
            print(f"\n{label}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-score:  {metrics['f1-score']:.3f}")
        else:
            print(f"\nAccuracy: {metrics:.3f}")


    return {
        "folds": fold_results,
        "summary": summary,
    }


def run_risk_cross_validation(risk_data_paths, create_dataset_fn, selected_model_name, args, device):
    """
    Run stratified CV for risk-only training.
    Emotion training is expected to be disabled in this mode.
    """
    cv_args = argparse.Namespace(**vars(args))
    cv_args.save_checkpoints = False
    cv_args.early_stopping = False

    risk_splits = {}
    for risk_name, path in risk_data_paths.items():
        rdf = pd.read_csv(path)
        rdf["label"] = rdf["label"].map(lambda x: 0 if "safe" in str(x).lower() else 1)
        _, test_df, folds = build_risk_cv_folds(rdf, cv_args.risk_cv_folds, cv_args.test_size)
        risk_splits[risk_name] = {"test_df": test_df, "folds": folds}

    fold_results = []

    for fold_idx in range(cv_args.risk_cv_folds):
        print(f"\n=== Risk CV Fold {fold_idx + 1}/{cv_args.risk_cv_folds} ===")

        fold_risk_loaders = {}
        for risk_name, split_info in risk_splits.items():
            fold_train_df = split_info["folds"][fold_idx]["train_df"]
            fold_val_df = split_info["folds"][fold_idx]["val_df"]
            fold_test_df = split_info["test_df"]

            fold_train_ds = create_dataset_fn(fold_train_df, use_risk=True, risk_name=risk_name)
            fold_val_ds = create_dataset_fn(fold_val_df, use_risk=True, risk_name=risk_name)
            fold_test_ds = create_dataset_fn(fold_test_df, use_risk=True, risk_name=risk_name)

            fold_risk_loaders[risk_name] = {
                "train": DataLoader(fold_train_ds, batch_size=max(1, cv_args.batch_size // 2), shuffle=True),
                "val": DataLoader(fold_val_ds, batch_size=max(1, cv_args.batch_size // 2), shuffle=False),
                "test": DataLoader(fold_test_ds, batch_size=max(1, cv_args.batch_size // 2), shuffle=False),
            }

        primary_risk_name = next(iter(fold_risk_loaders))
        primary_train_loader = fold_risk_loaders[primary_risk_name]["train"]
        primary_val_loader = fold_risk_loaders[primary_risk_name]["val"]

        fold_model = BertEmotionRiskModel(
            selected_model_name,
            num_labels=len(args._label2id),
            use_risk=True,
            dropout_rate=args.dropout_rate,
        ).to(device)

        total_steps = len(primary_train_loader) * cv_args.epochs
        warmup_steps = int(cv_args.warmup_ratio * total_steps)
        optimizer = create_optimizer(fold_model, lr=cv_args.learning_rate, weight_decay=0.05)
        scheduler = create_scheduler(optimizer, warmup_steps, total_steps)

        val_metrics, history, fold_risk_metrics, fold_risk_outputs = train_one_fold(
            fold_model,
            primary_train_loader,
            primary_val_loader,
            optimizer,
            scheduler,
            device,
            cv_args,
            epoch_start=0,
            train_df=None,
            risk_loaders=fold_risk_loaders,
        )

        test_risk_metrics, test_risk_outputs = validate_risk_datasets(
            fold_model,
            fold_risk_loaders,
            device,
            split="test",
            debug=cv_args.debug,
        )

        fold_result = {
            "fold": fold_idx + 1,
            "val_risk_metrics": fold_risk_metrics,
            "test_risk_metrics": test_risk_metrics,
        }
        fold_results.append(fold_result)

        print(f"Fold {fold_idx + 1} validation metrics:")
        for risk_name, rm in fold_risk_metrics.items():
            print(f"  {risk_name}: F1={rm['f1']:.3f}, AUROC={rm['auroc']:.3f}, Precision={rm['precision']:.3f}, Recall={rm['recall']:.3f}")

        print(f"Fold {fold_idx + 1} test metrics:")
        for risk_name, rm in test_risk_metrics.items():
            print(f"  {risk_name}: F1={rm['f1']:.3f}, AUROC={rm['auroc']:.3f}, Precision={rm['precision']:.3f}, Recall={rm['recall']:.3f}")

    summary = {}
    for risk_name in risk_splits.keys():
        summary[risk_name] = {
            "validation": summarize_metric_dicts(
                [fold_result["val_risk_metrics"][risk_name] for fold_result in fold_results]
            ),
            "test": summarize_metric_dicts(
                [fold_result["test_risk_metrics"][risk_name] for fold_result in fold_results]
            ),
        }

    print("\n=== Cross-Validation Summary ===")
    for risk_name, risk_summary in summary.items():
        test_summary = risk_summary["test"]
        if "f1" in test_summary:
            print(
                f"{risk_name} | test F1={test_summary['f1']['mean']:.3f} ± {test_summary['f1']['std']:.3f} | "
                f"AUROC={test_summary['auroc']['mean']:.3f} ± {test_summary['auroc']['std']:.3f}"
            )

    return {
        "folds": fold_results,
        "summary": summary,
    }


# =========================
# Refactored train_one_fold
# =========================

def train_one_fold(model, train_loader, val_loader, optimizer, scheduler,
                   device, args, epoch_start=0, train_df=None, risk_loaders=None):

    # Setup criterion with class weights
    if train_df is not None and args.use_emotions:
        label_counts = train_df["label"].value_counts().sort_index()
        total_samples = len(train_df)
        class_weights = total_samples / (len(label_counts) * label_counts.values)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)

        print("\nClass weights:")
        for i, (label_name, weight) in enumerate(zip(args._id2label.values(), class_weights)):
            print(f"  {label_name}: {weight:.2f}")
    elif args.use_emotions:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = None

    history = {"train_loss": [], "val_loss": [], "emotion_metrics": [], 
               "risk_metrics": [], "risk_threshold_sweeps": []}
    use_early_stopping = args.early_stopping and args.use_emotions
    if args.early_stopping and not args.use_emotions:
        print("Early stopping is disabled because emotion validation is turned off.")
    early_stopper = EarlyStopping(patience=args.patience) if use_early_stopping else None
    accumulation_steps = getattr(args, "accumulation_steps", 1)

    risk_iters = {name: iter(loaders["train"]) for name, loaders in risk_loaders.items()} if risk_loaders else None

    for epoch in range(epoch_start, args.epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0

        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")

        for step, batch in enumerate(tqdm(train_loader, desc="Training", leave=True)):
            batch_loss = train_batch(
                model=model,
                batch=batch,
                criterion=criterion,
                device=device,
                accumulation_steps=accumulation_steps,
                use_emotions=args.use_emotions,
                risk_iters=risk_iters,
                risk_loaders=risk_loaders,
                risk_loss_weight=args.risk_loss_weight
            )

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += batch_loss

        avg_train_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # ----- Validate emotions only -----
        if args.use_emotions:
            avg_val_loss, metrics = validate_emotions_only(model, val_loader, device, args)
        else:
            avg_val_loss, metrics = float("nan"), None
        history["val_loss"].append(avg_val_loss)
        history["emotion_metrics"].append(metrics)

        # ----- Validate risk datasets separately -----
        risk_metrics, risk_outputs = validate_risk_datasets(model, risk_loaders, device, debug=args.debug)
        history["risk_metrics"].append(risk_metrics)

        # Optional threshold sweep
        epoch_risk_threshold_sweeps = {}
        for risk_name, data in risk_outputs.items():
            sweep_metrics = threshold_sweep_metrics(labels=data["labels"], probs=data["probs"])
            epoch_risk_threshold_sweeps[risk_name] = sweep_metrics
        history["risk_threshold_sweeps"].append(epoch_risk_threshold_sweeps)

        val_loss_display = f"{avg_val_loss:.4f}" if args.use_emotions else "N/A (emotions off)"
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_display}")
        for risk_name, rm in risk_metrics.items():
            #print(f"{risk_name} | F1={rm['f1']:.3f}, AUROC={rm['auroc']:.3f}, Precision={rm['precision']:.3f}, Recall={rm['recall']:.3f}")
            # Add metric confidence flags to logs (diagnostic transparency)
            status = "OK" if rm["valid"] else "UNSTABLE"
            print(
                f"{risk_name} | F1={rm['f1']:.3f}, "
                f"AUROC={rm['auroc']:.3f}, "
                f"Precision={rm['precision']:.3f}, "
                f"Recall={rm['recall']:.3f} | "
                f"support={rm['support']}, "
                f"status={status}"
            )

        # Early stopping
        if use_early_stopping and early_stopper is not None:
            if early_stopper.step(avg_val_loss):
                print("Early stopping triggered.")
                break
        
        # ----- Save checkpoint (optional) -----
        if args.save_checkpoints:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model_version}_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    return metrics, history, risk_metrics, risk_outputs


# =========================
# Label mapping
# =========================

def build_label_mappings(df, label_option):
    if label_option == 28:
        label2id = {lab: idx for idx, lab in enumerate(GOEMOTIONS_28_ORDER)}
        id2label = {idx: lab for lab, idx in label2id.items()}
    else:
        unique_labels = sorted(df["label"].unique())
        label2id = {l: i for i, l in enumerate(unique_labels)}
        id2label = {i: l for l, i in label2id.items()}

    df = df.copy()
    df["label"] = df["label"].map(label2id).astype(int)
    return label2id, id2label, df


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    # ------------------------
    # General model/dataset args
    # ------------------------
    parser.add_argument("--labels", type=int, choices=[6, 7, 28], default=6)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--lexicon_path", type=str, default=DEFAULT_LEXICON_PATH_6)
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_EMOTION_MODEL_NAME,
        help="Model used when both emotions and risk flags are trained together"
    )
    parser.add_argument(
        "--emotion_model_name",
        type=str,
        default=DEFAULT_EMOTION_MODEL_NAME,
        help="Model used for emotion-only training"
    )
    parser.add_argument(
        "--risk_model_name",
        type=str,
        default=DEFAULT_RISK_MODEL_NAME,
        help="Model used for risk-only training"
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--risk_cv_folds", type=int, default=1, help="Number of stratified CV folds for risk-only training")
    parser.add_argument("--emotion_cv_folds", type=int, default=1, help="Number of stratified CV folds for emotion-only training")
    # ------------------------
    # Training options
    # ------------------------
    parser.add_argument("--use_lexicon", action="store_true")
    emotion_group = parser.add_mutually_exclusive_group()
    emotion_group.add_argument("--use_emotions", dest="use_emotions", action="store_true")
    emotion_group.add_argument("--off_emotions", dest="use_emotions", action="store_false")
    parser.set_defaults(use_emotions=True)
    parser.add_argument("--use_risk_flags", action="store_true")
    parser.add_argument("--risk_loss_weight", type=float, default=0.3)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to a checkpoint .pt file to resume training")
    parser.add_argument("--model_version", type=str, default="v1_0")
    parser.add_argument("--risk_data_paths", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    # ------------------------
    # Inference options
    # ------------------------

    parser.add_argument("--infer_text", type=str, default=None)
    parser.add_argument("--infer_text_file", type=str, default=None)
    parser.add_argument("--infer_max_length", type=int, default=None)
    parser.add_argument("--infer_overlap", type=int, default=20)

    args = parser.parse_args()
    if args.val_size <= 0 or args.test_size <= 0:
        raise ValueError("--val_size and --test_size must be > 0")
    if args.val_size + args.test_size >= 1.0:
        raise ValueError("--val_size + --test_size must be < 1.0")
    if not args.use_emotions and not args.use_risk_flags:
        raise ValueError("No task selected. Enable --use_emotions and/or --use_risk_flags.")
    if args.risk_cv_folds < 1:
        raise ValueError("--risk_cv_folds must be at least 1")
    if args.emotion_cv_folds < 1:
        raise ValueError("--emotion_cv_folds must be at least 1")
    

    # Resolve model name by active task mode
    if args.use_emotions and args.use_risk_flags:
        selected_model_name = args.model_name
        selected_mode = "joint (emotions+risk)"
    elif args.use_emotions:
        selected_model_name = args.emotion_model_name
        selected_mode = "emotion-only"
    else:
        selected_model_name = args.risk_model_name
        selected_mode = "risk-only"

    print(f"Task mode: {selected_mode}")
    print(f"Selected base model: {selected_model_name}")

    # ------------------------
    # Set default paths
    # ------------------------
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints_{args.model_version}"

    if args.data_path is None or args.lexicon_path is None:
        if args.labels == 7:
            args.data_path = DEFAULT_7_LABEL_PATH
            if args.lexicon_path == DEFAULT_LEXICON_PATH_6: # only override if it's the default
                args.lexicon_path = DEFAULT_LEXICON_PATH_7
        elif args.labels == 6:
            args.data_path = DEFAULT_6_LABEL_PATH
            args.lexicon_path = DEFAULT_LEXICON_PATH_6
        else:
            args.data_path = DEFAULT_28_LABEL_PATH
            if args.lexicon_path == DEFAULT_LEXICON_PATH_6: # only override if it's the default
                args.lexicon_path = DEFAULT_LEXICON_PATH_28

    # ------------------------
    # Load dataframe
    # ------------------------
    df = pd.read_csv(args.data_path)

    # rename label 'suprise' to 'surprise' (if still present typo — triple checked)
    df['label'] = df['label'].replace('suprise', 'surprise')

    # Debug mode: sample 100 random rows per each label from emotions dataset
    if args.debug:
        df = df.groupby('label').sample(n=min(100, len(df)), random_state=42)
        print(f"Debug mode: Using {len(df)} random samples from emotions dataset")
        args.model_version += "_debug"
    
    # Check label distribution
    print("Label distribution:")
    print(df["label"].value_counts())

    # Initialize lexicon and inferencer if needed
    if args.use_lexicon:
        lexicon = ThemeLexicon(args.lexicon_path)
    else:
        lexicon = None

    # Resolve risk dataset paths
    if args.use_risk_flags:
        if args.risk_data_paths:
            with open(args.risk_data_paths, "r") as f:
                risk_data_paths = json.load(f)
        else:
            risk_data_paths = DEFAULT_RISK_DATASETS
    else:
        risk_data_paths = None

    # ------------------------
    # Build label mappings
    # ------------------------
    label2id, id2label, df = build_label_mappings(df, args.labels)
    args._label2id = label2id
    args._id2label = id2label

    # ------------------------
    # Initialize tokenizer
    # ------------------------
    tokenizer = AutoTokenizer.from_pretrained(selected_model_name)
    tokenizer.model_max_length = args.max_length  # enforce max_length globally

    # ------------------------
    # Device setup
    # ------------------------
    device = get_device()
    # Clear GPU cache and limit memory if needed
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False  # stabilize memory usage
    torch.backends.cudnn.deterministic = True
    
    print(f"Device: {device}")  # Should print: device(type='cuda')

    if torch.cuda.is_available():
        # Monitor GPU memory
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # ------------------------
    # Split dataset (train / val / test)
    # ------------------------
    holdout_size = args.val_size + args.test_size
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_size, stratify=df["label"], random_state=42
    )
    test_frac_in_holdout = args.test_size / holdout_size
    val_df, test_df = train_test_split(
        holdout_df, test_size=test_frac_in_holdout, stratify=holdout_df["label"], random_state=42
    )

    print(
        f"Data split | train={len(train_df)} ({len(train_df)/len(df):.1%}) | "
        f"val={len(val_df)} ({len(val_df)/len(df):.1%}) | "
        f"test={len(test_df)} ({len(test_df)/len(df):.1%})"
    )

    # ------------------------
    # Create Datasets and DataLoaders
    # ------------------------
    def create_dataset(df_subset, use_risk=False, risk_name=None):
        return TextDataset(
            df_subset,
            tokenizer,
            args.max_length,
            args.labels,
            lexicon,
            args.use_lexicon,
            use_risk=use_risk,
            risk_name=risk_name
        )

    train_ds = create_dataset(train_df, use_risk=args.use_risk_flags)
    val_ds = create_dataset(val_df, use_risk=args.use_risk_flags)
    test_ds = create_dataset(test_df, use_risk=args.use_risk_flags)
    #train_ds = create_dataset(train_df, use_risk=False)
    #val_ds = create_dataset(val_df, use_risk=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    #train_loader = DataLoader(train_ds, batch_size=args.batch_size // 2, shuffle=True)
    #val_loader = DataLoader(val_ds, batch_size=args.batch_size // 2)
    # ------------------------
    # Risk flag loaders
    # ------------------------
    """risk_loaders = []

    if args.use_risk_flags:
        for risk_name, path in risk_data_paths.items():
            rdf = pd.read_csv(path)

            # Map labels: safe -> 0, risk -> 1
            rdf["label"] = rdf["label"].map(
                lambda x: 0 if "safe" in str(x).lower() else 1
            )

            rds = create_dataset(rdf, use_risk=True, risk_name=risk_name)
            risk_loaders.append(DataLoader(rds, batch_size=args.batch_size, shuffle=True))
        print(f"Loaded {len(risk_loaders)} risk datasets")"""

    """risk_loaders = []
    if args.use_risk_flags:
        for risk_name, path in risk_data_paths.items():
            rdf = pd.read_csv(path)
            rdf["label"] = rdf["label"].map(lambda x: 0 if "safe" in str(x).lower() else 1)
            #rds = create_dataset(rdf, use_risk=True, risk_name=risk_name)
            #risk_loaders.append((risk_name, DataLoader(rds, batch_size=args.batch_size//2)))

            # Make risk validation stratified
            rtrain_df, rval_df = train_test_split(rdf, test_size=0.3, stratify=rdf["label"], random_state=42)

            rtrain_ds = create_dataset(rtrain_df, use_risk=True, risk_name=risk_name)
            rval_ds   = create_dataset(rval_df,   use_risk=True, risk_name=risk_name)

            risk_loaders.append((risk_name,
                {
                    "train": DataLoader(rtrain_ds, batch_size=args.batch_size // 2, shuffle=True),
                    "val":   DataLoader(rval_ds,   batch_size=args.batch_size // 2, shuffle=False),
                }
            ))"""
    risk_loaders = {} 
    if args.use_risk_flags:
        for risk_name, path in risk_data_paths.items():
            rdf = pd.read_csv(path)
            # Map labels: safe -> 0, risk -> 1
            rdf["label"] = rdf["label"].map(lambda x: 0 if "safe" in str(x).lower() else 1)

            # stratified split (train / val / test)
            rholdout_size = args.val_size + args.test_size
            rtrain_df, rholdout_df = train_test_split(
                rdf, test_size=rholdout_size, stratify=rdf["label"], random_state=42
            )
            rtest_frac_in_holdout = args.test_size / rholdout_size
            rval_df, rtest_df = train_test_split(
                rholdout_df,
                test_size=rtest_frac_in_holdout,
                stratify=rholdout_df["label"],
                random_state=42
            )

            rtrain_ds = create_dataset(rtrain_df, use_risk=True, risk_name=risk_name)
            rval_ds   = create_dataset(rval_df,   use_risk=True, risk_name=risk_name)
            rtest_ds  = create_dataset(rtest_df,  use_risk=True, risk_name=risk_name)

            risk_loaders[risk_name] = {
                "train": DataLoader(rtrain_ds, batch_size=args.batch_size // 2, shuffle=True),
                "val":   DataLoader(rval_ds,   batch_size=args.batch_size // 2, shuffle=False),
                "test":  DataLoader(rtest_ds,  batch_size=args.batch_size // 2, shuffle=False)
            }
        if args.debug:
            # also shrink risk datasets
            #for risk_name in risk_loaders:
            #    risk_loaders[risk_name]["train"] = Subset(risk_loaders[risk_name]["train"].dataset, range(10))
            for risk_name in risk_loaders:
                subset = Subset(risk_loaders[risk_name]["train"].dataset, range(10))
                risk_loaders[risk_name]["train"] = DataLoader(
                    subset,
                    batch_size=2,   # small batch for debug
                    shuffle=True
                )

    # Risk-only CV mode: run stratified cross-validation and stop here.
    if args.use_risk_flags and not args.use_emotions and args.risk_cv_folds > 1:
        cv_results = run_risk_cross_validation(
            risk_data_paths=risk_data_paths,
            create_dataset_fn=create_dataset,
            selected_model_name=selected_model_name,
            args=args,
            device=device,
        )

        outdir = f"saved_models/trained_model_{args.model_version}"
        os.makedirs(outdir, exist_ok=True)
        cv_file = os.path.join(outdir, "risk_cv_summary.json")
        with open(cv_file, "w") as f:
            json.dump(cv_results, f, indent=4)
        print(f"Risk CV summary saved to {cv_file}")
        return
    # Emotion-only CV mode
    if args.use_emotions and not args.use_risk_flags and args.emotion_cv_folds > 1:
        cv_results = run_emotion_cross_validation(
            df=df,
            create_dataset_fn=create_dataset,
            selected_model_name=selected_model_name,
            args=args,
            device=device,
        )

        outdir = f"saved_models/trained_model_{args.model_version}"
        os.makedirs(outdir, exist_ok=True)
        cv_file = os.path.join(outdir, "emotion_cv_summary.json")
        with open(cv_file, "w") as f:
            json.dump(cv_results, f, indent=4)
        print(f"Emotion CV summary saved to {cv_file}")
        return
    # ------------------------
    # Initialize model
    # ------------------------
    for batch in train_loader:
        print(batch["input_ids"].shape)  # should be [batch_size, 128]
        break

    model = BertEmotionRiskModel(
        selected_model_name,
        num_labels=len(label2id),
        use_risk=args.use_risk_flags,
        dropout_rate=args.dropout_rate
    ).to(device)

    # VERIFY MODEL IS ON GPU
    print(f"Model device check: {next(model.parameters()).device}")
    print(f"Model initialized | emotions={len(label2id)} | use_risk={args.use_risk_flags}")
     # ------------------------
    # Optimizer & Scheduler
    # ------------------------
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)  # 6% warmup - default by arg --warmup_ratio

    optimizer = create_optimizer(model, lr=args.learning_rate, weight_decay=0.05) 
    scheduler = create_scheduler(optimizer, warmup_steps, total_steps)
        
    # ------------------------
    # Optional resume from checkpoint
    # ------------------------
    start_epoch = 0
    history = {}  # empty if starting fresh

    if args.resume_checkpoint:  # path to a .pt file
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("history", {})
        print(f"Resuming training from epoch {start_epoch}")

    # ------------------------
    # Train
    # ------------------------
    val_metrics, history, val_risk_metrics, val_risk_outputs = train_one_fold(
        model, train_loader, val_loader, optimizer, 
        scheduler, device, args, epoch_start=start_epoch, train_df=train_df, risk_loaders=risk_loaders
    )

    # ------------------------
    # Final held-out test evaluation
    # ------------------------
    if args.use_emotions:
        test_loss, test_metrics = validate_emotions_only(model, test_loader, device, args)
    else:
        test_loss, test_metrics = float("nan"), None
    test_risk_metrics, test_risk_outputs = validate_risk_datasets(
        model,
        risk_loaders,
        device,
        split="test",
        debug=args.debug
    )

    # ------------------------
    # Print results
    # ------------------------
    if args.use_emotions:
        print(f"\nFinal Test Loss: {test_loss:.4f}")
        print(test_metrics["report"])
    else:
        print("\nEmotion task disabled: skipped emotion test evaluation.")
    if args.use_risk_flags:
        print("\n=== Risk Test Metrics ===")
        for risk_name, rm in test_risk_metrics.items():
            print(f"{risk_name}: {rm}")
    # ------------------------
    # Save metrics & visualization
    # ------------------------
    outdir = f"saved_models/trained_model_{args.model_version}"
    os.makedirs(outdir, exist_ok=True)
    viz_dir = os.path.join(outdir, "metrics")
    os.makedirs(viz_dir, exist_ok=True)

    # Plot training history (loss)
    plot_train_history(
        history, save_to=os.path.join(viz_dir, "train_history.png")
    )
    # Plot confusion matrix
    if args.use_emotions:
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            label_names=list(id2label.values()),
            save_to=os.path.join(viz_dir, "confusion_matrix.png")
        )
        # Save classification report to a text file
        save_classification_report(
            test_metrics["report"], save_to=os.path.join(viz_dir, "classification_report.txt")
        )

    # Plot risk metrics if applicable
    if args.use_risk_flags:
        risk_viz_dir = os.path.join(viz_dir, "risk_metrics")
        plot_risk_metrics(test_risk_metrics, save_dir=risk_viz_dir)

    roc_dir = os.path.join(viz_dir, "roc_curves")
    os.makedirs(roc_dir, exist_ok=True)

    # Per-risk ROC
    for risk_name, data in test_risk_outputs.items():
        plot_risk_roc_curve(
            data["labels"],
            data["probs"],
            risk_name,
            save_to=os.path.join(roc_dir, f"roc_{risk_name}.png")
        )

    # Combined ROC
    if args.debug:
        print("Debug mode: skipping ROC and threshold sweep plots")
    else:
        plot_combined_risk_roc_curves(
            test_risk_outputs,
            save_to=os.path.join(roc_dir, "roc_all_risks.png")
        )

    # ------------------------
    # Threshold sweep plots
    # ------------------------
    threshold_dir = os.path.join(viz_dir, "threshold_sweeps")
    os.makedirs(threshold_dir, exist_ok=True)

    optimal_thresholds = {}

    for risk_name, data in test_risk_outputs.items():
        metrics_sweep = threshold_sweep_metrics(
            labels=data["labels"],
            probs=data["probs"]
        )
        opt_thr = plot_threshold_sweep(
            metrics_sweep,
            risk_name,
            save_to=os.path.join(threshold_dir, f"{risk_name}_threshold_sweep.png")
        )
        optimal_thresholds[risk_name] = float(opt_thr)

    # Save optimal thresholds
    threshold_file = os.path.join(threshold_dir, "optimal_thresholds.json")
    with open(threshold_file, "w") as f:
        json.dump(optimal_thresholds, f, indent=4)

    print(f"Optimal thresholds saved to {threshold_file}")

    # Combined dashboard figure
    plot_combined_dashboard( 
        history,
        risk_metrics_history=history["risk_metrics"],
        optimal_thresholds_history=history["risk_threshold_sweeps"],
        save_to=os.path.join(viz_dir, "combined_dashboard.png")
    )
    # ------------------------
    # Save final model
    # ------------------------
    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print(f"Final model saved to {outdir}")

    # ------------------------
    # Optional inference
    # ------------------------
    if args.infer_text or args.infer_text_file:
        text = args.infer_text or open(args.infer_text_file, "r", encoding="utf-8").read()
        predictions = predict_chunked(
            text, model, tokenizer, device, id2label,
            max_length=args.infer_max_length or args.max_length,
            overlap=args.infer_overlap
        )
        for pred in predictions:
            print(pred)


if __name__ == "__main__":
    main()

    # Optional: Check GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 GPU cache cleared")