# project-root/bert/train.py
import argparse
import pandas as pd
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from visualize_dashboard import plot_combined_dashboard
from dataset import TextDataset
from model_utils import create_optimizer, create_scheduler, EarlyStopping
from metrics import compute_all_metrics, compute_risk_metrics, threshold_sweep_metrics
from visualize_metrics import plot_combined_risk_roc_curves, plot_confusion_matrix, plot_risk_roc_curve, plot_threshold_sweep, plot_train_history, save_classification_report, plot_risk_metrics
from lexicon_utils import ThemeLexicon, SubthemeInferencer
from multitask_model import BertEmotionRiskModel

from inference import predict_chunked

# Uncomment this if previous line is commented out
#try:
#    from inference import predict_chunked
#except ImportError:
#    predict_chunked = None

DEFAULT_6_LABEL_PATH = "../data/processed/dataset_6labels_clean_more.csv"
DEFAULT_28_LABEL_PATH = "../data/processed/goemotions.csv" # to add later
DEFAULT_LEXICON_PATH_6 = "../data/lexicon/lexicon_clean_6.json"

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
                risk_iters=None, risk_loaders=None, risk_loss_weight=1.0):
    
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["emotion_labels"],
        risk_labels=None
    )

    loss = outputs.loss
    total_loss = loss

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

            total_loss = total_loss + risk_loss_weight * risk_loss

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


def validate_risk_datasets(model, risk_loaders, device, debug=False):
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
        r_loader = loaders["val"]
        all_logits, all_labels = [], []

        # Optional debug subset
        if debug:
            r_loader = DataLoader(
                list(r_loader.dataset)[:50],
                batch_size=r_loader.batch_size,
                shuffle=False
            )

        with torch.no_grad():
            for rbatch in tqdm(r_loader, desc=f"Risk Validation ({risk_name})", leave=True):
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


# =========================
# Refactored train_one_fold
# =========================

def train_one_fold(model, train_loader, val_loader, optimizer, scheduler,
                   device, args, epoch_start=0, train_df=None, risk_loaders=None):

    # Setup criterion with class weights
    if train_df is not None:
        label_counts = train_df["label"].value_counts().sort_index()
        total_samples = len(train_df)
        class_weights = total_samples / (len(label_counts) * label_counts.values)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)

        print("\nClass weights:")
        for i, (label_name, weight) in enumerate(zip(args._id2label.values(), class_weights)):
            print(f"  {label_name}: {weight:.2f}")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "emotion_metrics": [], 
               "risk_metrics": [], "risk_threshold_sweeps": []}
    early_stopper = EarlyStopping(patience=args.patience) if args.early_stopping else None
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
        avg_val_loss, metrics = validate_emotions_only(model, val_loader, device, args)
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

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
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
        if args.early_stopping and early_stopper is not None:
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
    parser.add_argument("--labels", type=int, choices=[6, 28], default=6)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--lexicon_path", type=str, default=DEFAULT_LEXICON_PATH_6)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    # ------------------------
    # Training options
    # ------------------------
    parser.add_argument("--use_lexicon", action="store_true")
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
    # ------------------------
    # Set default paths
    # ------------------------
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints_{args.model_version}"

    if args.data_path is None:
        args.data_path = DEFAULT_6_LABEL_PATH if args.labels == 6 else DEFAULT_28_LABEL_PATH

    # ------------------------
    # Load dataframe
    # ------------------------
    df = pd.read_csv(args.data_path)

    # rename label 'suprise' to 'surprise'
    df['label'] = df['label'].replace('suprise', 'surprise')

    # Debug mode: sample 500 random rows from emotions dataset
    if args.debug:
        df = df.sample(n=min(500, len(df)), random_state=42)
        print(f"Debug mode: Using {len(df)} random samples from emotions dataset")
    
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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
    # Split dataset
    # ------------------------
    train_df, val_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
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
    #train_ds = create_dataset(train_df, use_risk=False)
    #val_ds = create_dataset(val_df, use_risk=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
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

            # stratified split
            rtrain_df, rval_df = train_test_split(rdf, test_size=0.3, stratify=rdf["label"], random_state=42)

            rtrain_ds = create_dataset(rtrain_df, use_risk=True, risk_name=risk_name)
            rval_ds   = create_dataset(rval_df,   use_risk=True, risk_name=risk_name)

            risk_loaders[risk_name] = {
                "train": DataLoader(rtrain_ds, batch_size=args.batch_size // 2, shuffle=True),
                "val":   DataLoader(rval_ds,   batch_size=args.batch_size // 2, shuffle=False)
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
    # ------------------------
    # Initialize model
    # ------------------------
    for batch in train_loader:
        print(batch["input_ids"].shape)  # should be [batch_size, 128]
        break

    model = BertEmotionRiskModel(
        args.model_name,
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
    metrics, history, risk_metrics, risk_outputs = train_one_fold(
        model, train_loader, val_loader, optimizer, 
        scheduler, device, args, epoch_start=start_epoch, train_df=train_df, risk_loaders=risk_loaders
    )

    # ------------------------
    # Print results
    # ------------------------
    print(metrics["report"])
    if args.use_risk_flags:
        print("\n=== Risk Metrics ===")
        for risk_name, rm in risk_metrics.items():
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
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        label_names=list(id2label.values()),
        save_to=os.path.join(viz_dir, "confusion_matrix.png")
    )
    # Save classification report to a text file
    save_classification_report(
        metrics["report"], save_to=os.path.join(viz_dir, "classification_report.txt")
    )

    # Plot risk metrics if applicable
    if args.use_risk_flags:
        risk_viz_dir = os.path.join(viz_dir, "risk_metrics")
        plot_risk_metrics(risk_metrics, save_dir=risk_viz_dir)

    roc_dir = os.path.join(viz_dir, "roc_curves")
    os.makedirs(roc_dir, exist_ok=True)

    # Per-risk ROC
    for risk_name, data in risk_outputs.items():
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
            risk_outputs,
            save_to=os.path.join(roc_dir, "roc_all_risks.png")
        )

    # ------------------------
    # Threshold sweep plots
    # ------------------------
    threshold_dir = os.path.join(viz_dir, "threshold_sweeps")
    os.makedirs(threshold_dir, exist_ok=True)

    optimal_thresholds = {}

    for risk_name, data in risk_outputs.items():
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