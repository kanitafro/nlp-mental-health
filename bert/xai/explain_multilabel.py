# bert/xai/explain_multilabel.py

from pathlib import Path
from typing import Dict, List, Optional
import os
import importlib.util
import csv
import html
import sys

# Prevent TensorFlow imports
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import torch
import shap
from captum.attr import IntegratedGradients

XAI_DIR = Path(__file__).resolve().parent
BERT_DIR = XAI_DIR.parent
if str(BERT_DIR) not in sys.path:
    sys.path.insert(0, str(BERT_DIR))

# Monkey‑patch to avoid torchvision
_original_find_spec = importlib.util.find_spec
def _find_spec_without_torchvision(name, package=None):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _find_spec_without_torchvision

from transformers import AutoConfig, AutoModel, AutoTokenizer
from multitask_model import BertEmotionRiskModel

importlib.util.find_spec = _original_find_spec

# ============================================================
# GoEmotions 28‑label set
# ============================================================
LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# ============================================================
# Project Signature Colors (unchanged)
# ============================================================
def get_xai_colors(dark_mode=False):
    bg_color = "#333333" if dark_mode else "white"
    text_color = "white" if dark_mode else "black"
    pink_color = "#FEB2B4" if dark_mode else "#FF7F7F"
    yellow_color = "#FCD639" if dark_mode else "#F5D000"
    mercury_color = "#BEC7B9" if dark_mode else "#819774"
    orange_color = "#F29668" if dark_mode else "#D16D3B"
    butteryellow_color = "#FFE497" if dark_mode else "#FFD769"
    return {
        "bg": bg_color,
        "text": text_color,
        "pink": pink_color,
        "yellow": yellow_color,
        "mercury": mercury_color,
        "orange": orange_color,
        "butteryellow": butteryellow_color,
    }

# ============================================================
# Model loading (multi‑label, 28 classes)
# ============================================================
def load_model(
    checkpoint_path: str,
    model_name: str = "distilbert-base-uncased",
    device: Optional[str] = None,
    num_labels: int = 28,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    # ------------------------------------------------------------
    # FIX: Rename 'classifier' -> 'emotion_classifier' if needed
    # ------------------------------------------------------------
    # Check if the checkpoint uses 'classifier' but the model expects 'emotion_classifier'
    if "classifier.weight" in cleaned_state_dict and "emotion_classifier.weight" not in cleaned_state_dict:
        print("Detected 'classifier' keys in checkpoint – renaming to 'emotion_classifier'...")
        new_state_dict = {}
        for key, value in cleaned_state_dict.items():
            if key.startswith("classifier."):
                new_key = "emotion_classifier." + key[len("classifier."):]
            else:
                new_key = key
            new_state_dict[new_key] = value
        cleaned_state_dict = new_state_dict

    # Determine backbone (bert or distilbert) from state_dict keys
    if any("encoder.layer" in key for key in cleaned_state_dict):
        model_name = "bert-base-uncased"
    else:
        model_name = "distilbert-base-uncased"

    # Check if risk classifier is present (not used here)
    use_risk = any(key.startswith("risk_classifier.") for key in cleaned_state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    base_model = AutoModel.from_pretrained(
        model_name,
        config=config,
    )

    model = BertEmotionRiskModel(
        config=config,
        base_model=base_model,
        num_labels=num_labels,
        use_risk=use_risk,
        dropout_rate=0.1,
    )

    # Load with strict=True – now keys should match
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer, device

# ============================================================
# Prediction wrapper (returns sigmoid probabilities)
# ============================================================
class EmotionPredictor:
    def __init__(self, model, tokenizer, device, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def __call__(self, texts, batch_size: int = 32) -> np.ndarray:
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif isinstance(texts, tuple):
            texts = list(texts)
        texts = [str(t) for t in texts]
        if not texts:
            return np.empty((0, len(LABELS)), dtype=np.float32)

        all_probs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Multi‑label: sigmoid, not softmax
                probs = torch.sigmoid(outputs.logits)
            all_probs.append(probs.detach().cpu().numpy())
            del encoded, outputs, probs
        return np.concatenate(all_probs, axis=0)

# ============================================================
# SHAP explainer
# ============================================================
def create_explainer(predictor: EmotionPredictor):
    masker = shap.maskers.Text(tokenizer=predictor.tokenizer)
    explainer = shap.Explainer(
        predictor,
        masker,
        output_names=LABELS,
    )
    return explainer

def explain_text(explainer, text: str, max_evals: int = 500):
    if not isinstance(text, str):
        text = str(text)
    shap_values = explainer([text], max_evals=max_evals)
    return shap_values

# ============================================================
# Prediction info (not used for selection, but kept for completeness)
# ============================================================
def get_prediction(predictor: EmotionPredictor, text: str) -> Dict:
    probs = predictor([text])[0]
    return {
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
    }

# ============================================================
# SHAP HTML visualization (unchanged, works for any emotion index)
# ============================================================
def _attribution_background(value, max_abs, positive_color, negative_color):
    if max_abs == 0:
        return "transparent"
    normalized = abs(value) / max_abs
    opacity = 0.12 + 0.68 * min(normalized, 1.0)
    if value >= 0:
        return f"rgba({int(positive_color[1:3],16)},{int(positive_color[3:5],16)},{int(positive_color[5:7],16)},{opacity:.3f})"
    return f"rgba({int(negative_color[1:3],16)},{int(negative_color[3:5],16)},{int(negative_color[5:7],16)},{opacity:.3f})"

def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def save_shap_html(
    shap_values,
    output_path: str,
    emotion_index: int,
    dark_mode: bool = False,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = get_xai_colors(dark_mode)
    bg_color = colors["bg"]
    text_color = colors["text"]
    positive_color = colors["pink"]
    negative_color = colors["yellow"]

    explanation = shap_values[0]
    tokens = explanation.data
    values = np.asarray(explanation.values)

    if values.ndim == 1:
        raise ValueError("SHAP returned single‑output, but multi‑label expected.")
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError(f"Unexpected SHAP shape {values.shape} – expected [tokens, {len(LABELS)}].")

    emotion = LABELS[emotion_index]
    emotion_values = values[:, emotion_index]
    max_abs = np.max(np.abs(emotion_values)) or 1.0

    token_html = []
    for token, val in zip(tokens, emotion_values):
        val = float(val)
        bg = _attribution_background(val, max_abs, positive_color, negative_color)
        token_html.append(
            f'<span title="SHAP: {val:+.6f}" style="background:{bg};padding:5px 7px;margin:3px;'
            f'display:inline-block;border-radius:5px;border:1px solid rgba(128,128,128,0.25);">'
            f'{html.escape(str(token))}</span>'
        )

    document = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>SHAP Explanation - {html.escape(emotion)}</title>
<style>
body {{ background-color:{bg_color}; color:{text_color}; font-family:Arial,sans-serif; margin:40px; }}
h1 {{ margin-bottom:5px; }}
.subtitle {{ opacity:0.75; }}
.legend {{ margin:20px 0; }}
.legend-item {{ display:inline-block; margin-right:25px; }}
.legend-box {{ display:inline-block; width:18px; height:18px; border-radius:4px; vertical-align:middle; margin-right:6px; }}
.text-container {{ font-size:19px; line-height:2.4; margin-top:25px; }}
.info {{ margin-top:30px; padding:15px; border-radius:8px; background:{_hex_to_rgba(colors['mercury'],0.15)}; }}
</style>
</head>
<body>
<h1>SHAP Token Attribution</h1>
<div class="subtitle">Emotion explained: <strong>{html.escape(emotion)}</strong></div>
<div class="legend">
<div class="legend-item"><span class="legend-box" style="background:{positive_color};"></span> Positive contribution</div>
<div class="legend-item"><span class="legend-box" style="background:{negative_color};"></span> Negative contribution</div>
</div>
<div class="text-container">{" ".join(token_html)}</div>
<div class="info"><strong>Interpretation:</strong> Pink = supports the emotion, Yellow = suppresses it.</div>
</body></html>"""
    output_path.write_text(document, encoding="utf-8")

# ============================================================
# Save token contributions (CSV)
# ============================================================
def save_token_contributions(shap_values, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    explanation = shap_values[0]
    tokens = explanation.data
    values = np.asarray(explanation.values)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "emotion", "shap_value"])
        for i, token in enumerate(tokens):
            for j, emotion in enumerate(LABELS):
                writer.writerow([token, emotion, float(values[i, j])])

# ============================================================
# Top tokens per emotion
# ============================================================
def get_top_tokens(shap_values, emotion: str, top_k: int = 10):
    if emotion not in LABELS:
        raise ValueError(f"Unknown emotion: {emotion}")
    idx = LABELS.index(emotion)
    explanation = shap_values[0]
    tokens = explanation.data
    values = np.asarray(explanation.values)[:, idx]
    pos_indices = np.argsort(values)[::-1]
    neg_indices = np.argsort(values)
    positive = [{"token": str(tokens[i]), "shap_value": float(values[i])}
                for i in pos_indices[:top_k] if values[i] > 0]
    negative = [{"token": str(tokens[i]), "shap_value": float(values[i])}
                for i in neg_indices[:top_k] if values[i] < 0]
    return {"emotion": emotion, "positive": positive, "negative": negative}

# ============================================================
# Integrated Gradients
# ============================================================
class IntegratedGradientsExplainer:
    def __init__(self, model, tokenizer, device, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.embedding_layer = model.get_input_embeddings()
        self.ig = IntegratedGradients(self.forward_func)

    def forward_func(self, inputs_embeds, attention_mask, target):
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits[:, target]   # logit for one label

    def explain(self, text, target_emotion_index, n_steps=50):
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        input_embeds = self.embedding_layer(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)

        attributions = self.ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask, target_emotion_index),
            n_steps=n_steps,
        )
        token_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        token_ids = input_ids.squeeze(0).detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return {
            "text": text,
            "tokens": tokens,
            "attributions": token_attributions,
            "emotion": LABELS[target_emotion_index],
            "emotion_index": target_emotion_index,
        }

def save_ig_token_contributions(ig_result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "integrated_gradient"])
        for token, attr in zip(ig_result["tokens"], ig_result["attributions"]):
            writer.writerow([token, float(attr)])

def save_ig_html(ig_result, output_path, dark_mode=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = get_xai_colors(dark_mode)
    bg_color = colors["bg"]
    text_color = colors["text"]
    positive_color = colors["pink"]
    negative_color = colors["yellow"]

    tokens = ig_result["tokens"]
    attributions = ig_result["attributions"]
    emotion = ig_result["emotion"]
    max_abs = max(abs(float(x)) for x in attributions) or 1.0

    token_html = []
    for token, val in zip(tokens, attributions):
        val = float(val)
        bg = _attribution_background(val, max_abs, positive_color, negative_color)
        token_html.append(
            f'<span title="IG: {val:+.6f}" style="background:{bg};padding:5px 7px;margin:3px;'
            f'display:inline-block;border-radius:5px;border:1px solid rgba(128,128,128,0.25);">'
            f'{html.escape(str(token))}</span>'
        )

    document = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Integrated Gradients - {html.escape(emotion)}</title>
<style>
body {{ background-color:{bg_color}; color:{text_color}; font-family:Arial,sans-serif; margin:40px; }}
h1 {{ margin-bottom:5px; }}
.subtitle {{ opacity:0.75; }}
.legend {{ margin:20px 0; }}
.legend-item {{ display:inline-block; margin-right:25px; }}
.legend-box {{ display:inline-block; width:18px; height:18px; border-radius:4px; vertical-align:middle; margin-right:6px; }}
.text-container {{ font-size:19px; line-height:2.4; margin-top:25px; }}
.info {{ margin-top:30px; padding:15px; border-radius:8px; background:{_hex_to_rgba(colors['mercury'],0.15)}; }}
</style>
</head>
<body>
<h1>Integrated Gradients</h1>
<div class="subtitle">Emotion explained: <strong>{html.escape(emotion)}</strong></div>
<div class="legend">
<div class="legend-item"><span class="legend-box" style="background:{positive_color};"></span> Positive contribution</div>
<div class="legend-item"><span class="legend-box" style="background:{negative_color};"></span> Negative contribution</div>
</div>
<div class="text-container">{" ".join(token_html)}</div>
<div class="info"><strong>Interpretation:</strong> Pink = increases the output, Yellow = decreases it.</div>
</body></html>"""
    output_path.write_text(document, encoding="utf-8")