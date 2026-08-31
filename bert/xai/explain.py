# bert/XAI/explain.py

from pathlib import Path
from typing import Dict, List, Optional
import os
import importlib.util
import csv
import html
import sys

# Transformers will otherwise try to import TensorFlow / Keras when SHAP probes
# the model type. This repo only needs the PyTorch path.
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

_original_find_spec = importlib.util.find_spec


def _find_spec_without_torchvision(name, package=None):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = _find_spec_without_torchvision

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from multitask_model import BertEmotionRiskModel

importlib.util.find_spec = _original_find_spec


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "love",
    "sadness",
    "surprise",
]


# ============================================================
# Project Signature Colors
# ============================================================

def get_xai_colors(dark_mode=False):
    """
    Return the project's signature colors.

    These colors match the palette defined in
    bert/visualize_metrics.py.
    """

    bg_color = "#333333" if dark_mode else "white"
    text_color = "white" if dark_mode else "black"

    # Main project colors
    pink_color = (
        "#FEB2B4"
        if dark_mode
        else "#FF7F7F"
    )

    yellow_color = (
        "#FCD639"
        if dark_mode
        else "#F5D000"
    )

    # Secondary project colors
    mercury_color = (
        "#BEC7B9"
        if dark_mode
        else "#819774"
    )

    orange_color = (
        "#F29668"
        if dark_mode
        else "#D16D3B"
    )

    butteryellow_color = (
        "#FFE497"
        if dark_mode
        else "#FFD769"
    )

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
# Model loading
# ============================================================

def load_model(
    checkpoint_path: str,
    model_name: str = MODEL_NAME,
    device: Optional[str] = None,
):
    """
    Load the Phase 1 DistilBERT emotion classifier.

    The checkpoint is expected to contain the model state used
    by AutoModelForSequenceClassification.
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # The checkpoint stores custom module prefixes such as `bert.*`, but the
    # actual backbone is determined by the inner layer layout.
    if any("encoder.layer" in key for key in cleaned_state_dict):
        model_name = "bert-base-uncased"
    else:
        model_name = "distilbert-base-uncased"

    use_risk = any(
        key.startswith("risk_classifier.")
        for key in cleaned_state_dict
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABELS),
    )

    base_model = AutoModel.from_pretrained(
        model_name,
        config=config,
    )

    model = BertEmotionRiskModel(
        config=config,
        base_model=base_model,
        num_labels=len(LABELS),
        use_risk=use_risk,
        dropout_rate=0.1,
    )

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


# ============================================================
# Prediction wrapper
# ============================================================

class EmotionPredictor:
    """
    SHAP-compatible prediction wrapper.

    Input:
        List[str]

    Output:
        numpy array of shape:
            [batch_size, 7]

    Values are softmax probabilities.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_length: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def __call__(
        self,
        texts,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        SHAP-compatible prediction wrapper.
    
        Accepts:
            str
            list[str]
            tuple[str]
            numpy.ndarray containing strings
    
        Returns:
            numpy array of shape [batch_size, 7]
        """
    
        # --------------------------------------------------------
        # Normalize SHAP / NumPy inputs
        # --------------------------------------------------------
    
        if isinstance(texts, str):
            texts = [texts]
    
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
    
        elif isinstance(texts, tuple):
            texts = list(texts)
    
        # Ensure we always have a flat list of Python strings
        texts = [
            str(text)
            for text in texts
        ]
    
        if len(texts) == 0:
            return np.empty(
                (0, len(LABELS)),
                dtype=np.float32,
            )
    
        all_probabilities = []
    
        # --------------------------------------------------------
        # Batch prediction
        # --------------------------------------------------------
    
        for start in range(
            0,
            len(texts),
            batch_size,
        ):
    
            batch_texts = texts[
                start:start + batch_size
            ]
    
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
    
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }
    
            with torch.no_grad():
    
                outputs = self.model(
                    **encoded
                )
    
                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1,
                )
    
            all_probabilities.append(
                probabilities.detach()
                .cpu()
                .numpy()
            )
    
            del encoded
            del outputs
            del probabilities
    
        return np.concatenate(
            all_probabilities,
            axis=0,
        )


# ============================================================
# SHAP explainer
# ============================================================

def create_explainer(
    predictor: EmotionPredictor,
):
    """
    Create a SHAP text explainer around the
    HuggingFace tokenizer/model prediction function.
    """

    masker = shap.maskers.Text(
        tokenizer=predictor.tokenizer
    )

    explainer = shap.Explainer(
        predictor,
        masker,
        output_names=LABELS,
    )

    return explainer


# ============================================================
# Explain one text with SHAP
# ============================================================

def explain_text(
    explainer,
    text: str,
    max_evals: int = 500,
):
    """
    Generate a SHAP explanation for one text.

    The text is passed as a single-element Python list
    because the SHAP text masker expects batched text input.
    """

    if not isinstance(text, str):
        text = str(text)

    shap_values = explainer(
        [text],
        max_evals=max_evals,
    )

    return shap_values


# ============================================================
# Extract prediction information
# ============================================================

def get_prediction(
    predictor: EmotionPredictor,
    text: str,
) -> Dict:

    probabilities = predictor([text])[0]

    ranked_indices = np.argsort(
        probabilities
    )[::-1]

    ranked_predictions = [
        {
            "label": LABELS[index],
            "probability": float(
                probabilities[index]
            ),
        }
        for index in ranked_indices
    ]

    return {
        "predicted_label": LABELS[
            ranked_indices[0]
        ],
        "predicted_probability": float(
            probabilities[ranked_indices[0]]
        ),
        "probabilities": {
            LABELS[i]: float(
                probabilities[i]
            )
            for i in range(len(LABELS))
        },
        "ranked_predictions": ranked_predictions,
    }


# ============================================================
# Print prediction
# ============================================================

def print_prediction(
    prediction: Dict,
):
    print("\nPrediction")
    print("-" * 60)

    print(
        f"Predicted emotion: "
        f"{prediction['predicted_label']}"
    )

    print(
        f"Confidence: "
        f"{prediction['predicted_probability']:.4f}"
    )

    print(
        "\nAll emotion probabilities:"
    )

    for item in prediction[
        "ranked_predictions"
    ]:

        print(
            f"  {item['label']:10s} "
            f"{item['probability']:.4f}"
        )


# ============================================================
# SHAP HTML visualization
# ============================================================

def _attribution_background(
    value,
    max_abs,
    positive_color,
    negative_color,
):
    """
    Convert an attribution value into a background color.

    Positive = project pink
    Negative = project yellow
    """

    if max_abs == 0:
        return "transparent"

    normalized = abs(value) / max_abs

    # Keep opacity readable
    opacity = (
        0.12
        + 0.68 * min(normalized, 1.0)
    )

    if value >= 0:

        return (
            f"rgba("
            f"{int(positive_color[1:3], 16)},"
            f"{int(positive_color[3:5], 16)},"
            f"{int(positive_color[5:7], 16)},"
            f"{opacity:.3f})"
        )

    return (
        f"rgba("
        f"{int(negative_color[1:3], 16)},"
        f"{int(negative_color[3:5], 16)},"
        f"{int(negative_color[5:7], 16)},"
        f"{opacity:.3f})"
    )


def _hex_to_rgba(
    hex_color,
    alpha,
):
    """
    Convert #RRGGBB to rgba(...).
    """

    hex_color = hex_color.lstrip("#")

    r = int(
        hex_color[0:2],
        16,
    )

    g = int(
        hex_color[2:4],
        16,
    )

    b = int(
        hex_color[4:6],
        16,
    )

    return (
        f"rgba({r}, {g}, {b}, {alpha})"
    )


def save_shap_html(
    shap_values,
    output_path: str,
    emotion_index: Optional[int] = None,
    dark_mode: bool = False,
):
    """
    Save a custom SHAP token-level HTML visualization.

    The visualization follows the project's signature colors.

    Positive attribution:
        Pink

    Negative attribution:
        Yellow
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    colors = get_xai_colors(
        dark_mode
    )

    bg_color = colors["bg"]
    text_color = colors["text"]
    positive_color = colors["pink"]
    negative_color = colors["yellow"]

    explanation = shap_values[0]

    tokens = explanation.data

    values = np.asarray(
        explanation.values
    )
    
    if values.ndim == 1:
        raise ValueError(
            "SHAP returned a single-output explanation, "
            "but Phase 1 expects 7 emotion outputs."
        )
    
    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP value shape: {values.shape}. "
            f"Expected [tokens, emotions]."
        )
    
    if values.shape[1] != len(LABELS):
        raise ValueError(
            f"Expected {len(LABELS)} SHAP outputs, "
            f"got {values.shape[1]}."
        )

    # --------------------------------------------------------
    # Select emotion
    # --------------------------------------------------------

    if emotion_index is None:
        raise ValueError(
            "emotion_index must be specified for Phase 1 "
            "multi-class emotion explanations."
        )

    emotion = LABELS[
        emotion_index
    ]

    # [tokens, emotions]
    emotion_values = values[
        :,
        emotion_index,
    ]

    max_abs = np.max(
        np.abs(emotion_values)
    )

    if max_abs == 0:
        max_abs = 1.0

    # --------------------------------------------------------
    # Token HTML
    # --------------------------------------------------------

    token_html = []

    for token, value in zip(
        tokens,
        emotion_values,
    ):

        value = float(value)

        background = _attribution_background(
            value=value,
            max_abs=max_abs,
            positive_color=positive_color,
            negative_color=negative_color,
        )

        token_html.append(
            f"""
            <span
                title="SHAP: {value:+.6f}"
                style="
                    background:{background};
                    padding:5px 7px;
                    margin:3px;
                    display:inline-block;
                    border-radius:5px;
                    border:1px solid
                        rgba(128,128,128,0.25);
                "
            >
                {html.escape(str(token))}
            </span>
            """
        )

    # --------------------------------------------------------
    # HTML document
    # --------------------------------------------------------

    document = f"""
    <!DOCTYPE html>
    <html>
    <head>
    
    <meta charset="UTF-8">
    
    <title>
    SHAP Explanation - {html.escape(emotion)}
    </title>
    
    <style>
    
    body {{
        background-color: {bg_color};
        color: {text_color};
        font-family: Arial, sans-serif;
        margin: 40px;
    }}
    
    h1 {{
        margin-bottom: 5px;
    }}
    
    .subtitle {{
        color: {text_color};
        opacity: 0.75;
    }}
    
    .legend {{
        margin: 20px 0;
    }}
    
    .legend-item {{
        display: inline-block;
        margin-right: 25px;
    }}
    
    .legend-box {{
        display: inline-block;
        width: 18px;
        height: 18px;
        border-radius: 4px;
        vertical-align: middle;
        margin-right: 6px;
    }}
    
    .text-container {{
        font-size: 19px;
        line-height: 2.4;
        margin-top: 25px;
    }}
    
    .info {{
        margin-top: 30px;
        padding: 15px;
        border-radius: 8px;
        background: {_hex_to_rgba(
            colors["mercury"],
            0.15
        )};
    }}
    
    </style>
    
    </head>
    
    <body>
    
    <h1>
    SHAP Token Attribution
    </h1>
    
    <div class="subtitle">
    Emotion explained:
    <strong>
    {html.escape(emotion)}
    </strong>
    </div>
    
    <div class="legend">
    
    <div class="legend-item">
    
    <span
    class="legend-box"
    style="background:{positive_color};"
    ></span>
    
    Positive contribution
    
    </div>
    
    <div class="legend-item">
    
    <span
    class="legend-box"
    style="background:{negative_color};"
    ></span>
    
    Negative contribution
    
    </div>
    
    </div>
    
    <div class="text-container">
    
    {" ".join(token_html)}
    
    </div>
    
    <div class="info">
    
    <strong>Interpretation:</strong>
    
    Tokens with stronger pink shading contribute
    more positively toward the selected emotion.
    Tokens with stronger yellow shading contribute
    against the selected emotion.
    
    </div>
    
    </body>
    </html>
    """

    output_path.write_text(
        document,
        encoding="utf-8",
    )


# ============================================================
# Save SHAP token contributions
# ============================================================

def save_token_contributions(
    shap_values,
    output_path: str,
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    explanation = shap_values[0]

    tokens = explanation.data

    values = np.asarray(
        explanation.values
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "token",
                "emotion",
                "shap_value",
            ]
        )

        for token_index, token in enumerate(
            tokens
        ):

            for emotion_index, emotion in enumerate(
                LABELS
            ):

                value = values[
                    token_index,
                    emotion_index,
                ]

                writer.writerow(
                    [
                        token,
                        emotion,
                        float(value),
                    ]
                )


# ============================================================
# Utility: strongest SHAP tokens
# ============================================================

def get_top_tokens(
    shap_values,
    emotion: str,
    top_k: int = 10,
):

    if emotion not in LABELS:

        raise ValueError(
            f"Unknown emotion: {emotion}. "
            f"Expected one of {LABELS}"
        )

    emotion_index = LABELS.index(
        emotion
    )

    explanation = shap_values[0]

    tokens = explanation.data

    values = np.asarray(
        explanation.values
    )[:, emotion_index]

    positive_indices = np.argsort(
        values
    )[::-1]

    negative_indices = np.argsort(
        values
    )

    positive = [
        {
            "token": str(tokens[i]),
            "shap_value": float(
                values[i]
            ),
        }
        for i in positive_indices[:top_k]
        if values[i] > 0
    ]

    negative = [
        {
            "token": str(tokens[i]),
            "shap_value": float(
                values[i]
            ),
        }
        for i in negative_indices[:top_k]
        if values[i] < 0
    ]

    return {
        "emotion": emotion,
        "positive": positive,
        "negative": negative,
    }


# ============================================================
# Integrated Gradients Explainer
# ============================================================

class IntegratedGradientsExplainer:
    """
    Integrated Gradients for the Phase 1 DistilBERT model.

    IG is applied to continuous input embeddings rather than
    discrete token IDs.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_length=128,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        self.embedding_layer = (
            self.model.get_input_embeddings()
        )

        self.ig = IntegratedGradients(
            self.forward_func
        )

    def forward_func(
        self,
        inputs_embeds,
        attention_mask,
        target,
    ):
        """
        Forward function returning the logit for
        one selected emotion.
        """

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        logits = outputs.logits

        return logits[:, target]

    def explain(
        self,
        text,
        target_emotion_index,
        n_steps=50,
    ):
        """
        Calculate Integrated Gradients for one text
        and one target emotion.
        """

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoded[
            "input_ids"
        ].to(self.device)

        attention_mask = encoded[
            "attention_mask"
        ].to(self.device)

        # ----------------------------------------------------
        # Continuous token embeddings
        # ----------------------------------------------------

        input_embeds = (
            self.embedding_layer(
                input_ids
            )
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline_embeds = torch.zeros_like(
            input_embeds
        )

        # ----------------------------------------------------
        # Integrated Gradients
        # ----------------------------------------------------

        attributions = self.ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(
                attention_mask,
                target_emotion_index,
            ),
            n_steps=n_steps,
        )

        # ----------------------------------------------------
        # Collapse embedding dimension
        # ----------------------------------------------------

        token_attributions = (
            attributions.sum(dim=-1)
        )

        token_attributions = (
            token_attributions
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        token_ids = (
            input_ids
            .squeeze(0)
            .detach()
            .cpu()
            .tolist()
        )

        tokens = (
            self.tokenizer.convert_ids_to_tokens(
                token_ids
            )
        )

        return {
            "text": text,
            "tokens": tokens,
            "attributions": token_attributions,
            "emotion": LABELS[
                target_emotion_index
            ],
            "emotion_index": target_emotion_index,
        }


# ============================================================
# Save IG token contributions
# ============================================================

def save_ig_token_contributions(
    ig_result,
    output_path,
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "token",
                "integrated_gradient",
            ]
        )

        for token, attribution in zip(
            ig_result["tokens"],
            ig_result["attributions"],
        ):

            writer.writerow(
                [
                    token,
                    float(attribution),
                ]
            )


# ============================================================
# Save IG HTML visualization
# ============================================================

def save_ig_html(
    ig_result,
    output_path,
    dark_mode=False,
):
    """
    Save Integrated Gradients visualization.

    Uses project signature colors:

        Pink   = positive contribution
        Yellow = negative contribution
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    colors = get_xai_colors(
        dark_mode
    )

    bg_color = colors["bg"]
    text_color = colors["text"]
    positive_color = colors["pink"]
    negative_color = colors["yellow"]

    tokens = ig_result["tokens"]

    attributions = (
        ig_result["attributions"]
    )

    emotion = ig_result["emotion"]

    max_abs = max(
        abs(float(x))
        for x in attributions
    )

    if max_abs == 0:
        max_abs = 1.0

    token_html = []

    for token, attribution in zip(
        tokens,
        attributions,
    ):

        value = float(attribution)

        background = _attribution_background(
            value=value,
            max_abs=max_abs,
            positive_color=positive_color,
            negative_color=negative_color,
        )

        token_html.append(
            f"""
            <span
                title="IG: {value:+.6f}"
                style="
                    background:{background};
                    padding:5px 7px;
                    margin:3px;
                    display:inline-block;
                    border-radius:5px;
                    border:1px solid
                        rgba(128,128,128,0.25);
                "
            >
                {html.escape(str(token))}
            </span>
            """
        )

    document = f"""
<!DOCTYPE html>
<html>
<head>

<meta charset="UTF-8">

<title>
Integrated Gradients - {html.escape(emotion)}
</title>

<style>

body {{
    background-color: {bg_color};
    color: {text_color};
    font-family: Arial, sans-serif;
    margin: 40px;
}}

h1 {{
    margin-bottom: 5px;
}}

.subtitle {{
    opacity: 0.75;
}}

.legend {{
    margin: 20px 0;
}}

.legend-item {{
    display: inline-block;
    margin-right: 25px;
}}

.legend-box {{
    display: inline-block;
    width: 18px;
    height: 18px;
    border-radius: 4px;
    vertical-align: middle;
    margin-right: 6px;
}}

.text-container {{
    font-size: 19px;
    line-height: 2.4;
    margin-top: 25px;
}}

.info {{
    margin-top: 30px;
    padding: 15px;
    border-radius: 8px;
    background: {_hex_to_rgba(
        colors["mercury"],
        0.15
    )};
}}

</style>

</head>

<body>

<h1>
Integrated Gradients
</h1>

<div class="subtitle">
Emotion explained:
<strong>
{html.escape(emotion)}
</strong>
</div>

<div class="legend">

<div class="legend-item">

<span
class="legend-box"
style="background:{positive_color};"
></span>

Positive contribution

</div>

<div class="legend-item">

<span
class="legend-box"
style="background:{negative_color};"
></span>

Negative contribution

</div>

</div>

<div class="text-container">

{" ".join(token_html)}

</div>

<div class="info">

<strong>Interpretation:</strong>

Positive attribution (pink) means that the token increases
the model's output for the selected emotion.
Negative attribution (yellow) means that the token decreases
the model's output for the selected emotion.

</div>

</body>
</html>
"""

    output_path.write_text(
        document,
        encoding="utf-8",
    )