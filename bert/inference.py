# project-root/bert/inference.py
import re
import json
from unittest import result
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer
from typing import List, Dict, Optional

# lexicon-based subtheme inference (post-processing)
from lexicon_utils import ThemeLexicon, SubthemeInferencer

T = 1.5 # temperature for softmax

def count_ngrams(text: str, ngrams: List[str]) -> int:
    """
    Counts how many distinct n-grams from the list appear in the text.
    Case-insensitive, substring-based (conservative).
    """
    text = text.lower()
    count = 0
    for ng in ngrams:
        if ng.lower() in text:
            count += 1
    return count


def predict_chunked(
    text,
    model,
    tokenizer,
    device,
    id2label,
    max_length=128,
    overlap=20,
    batch_size=8,
):
    """
    Split long text into overlapping chunks and run inference on each chunk.
    Returns aggregated predictions.
    """
    # Tokenize the whole text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    start = 0
    
    # Create overlapping chunks
    while start < len(tokens):
        end = start + max_length
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
        start = end - overlap
    
    # Run inference on each chunk
    all_predictions = []
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        with torch.no_grad():
            enc = tokenizer(
                batch_chunks,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            for j in range(len(batch_chunks)):
                pred_idx = torch.argmax(probs[j]).item()
                confidence = probs[j][pred_idx].item()
                emotion = id2label[pred_idx]
                
                all_predictions.append({
                    "chunk": batch_chunks[j],
                    "emotion": emotion,
                    "confidence": confidence,
                    "all_probs": probs[j].cpu().numpy()
                })
    
    return all_predictions

def extract_textual_grounding(
    text: str,
    risk_ngrams: List[str],
    max_matches: int = 5
) -> List[str]:
    """
    Extracts surface-level n-grams from the text that match
    curated risk-specific patterns.

    - Case-insensitive
    - Substring-based
    - Deduplicated
    - NEVER affects prediction logic
    """
    text_lower = text.lower()
    matches = []

    for ng in risk_ngrams:
        ng_lower = ng.lower()
        if ng_lower in text_lower:
            matches.append(ng)
        if len(matches) >= max_matches:
            break

    return matches

@torch.no_grad()
def run_inference(
    model,
    texts: List[str],
    tokenizer_name: str,
    device: str = "cuda",
    max_length: int = 128,
    use_risk: bool = False,
    use_subthemes: bool = False,
    lexicon_path: Optional[str] = None,
    emotion_names: Optional[List[str]] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    optimal_thresholds_path: Optional[str] = None,
    risk_ngrams_path: Optional[str] = None,  
) -> Dict:
    """
    Runs inference for emotion classification (+ optional risk flags)
    and optional lexicon-based subtheme inference.

    Returns:
        {
            "emotion_logits": Tensor [N, num_labels],
            "emotion_probs": Tensor [N, num_labels],
            "risk_logits": Tensor [N, 4] (optional),
            "risk_probs": Tensor [N, 4] (optional),
            "subthemes": List[List[Dict]] (optional)
        }
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}
    model = model.to(device)
    model.eval()

    outputs = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"]
    )

    emotion_logits = outputs.logits
    emotion_probs = F.softmax(emotion_logits/T, dim=-1)

    result = {
        "emotion_logits": emotion_logits.cpu(),
        "emotion_probs": emotion_probs.cpu(),
    }

    # --------------------
    # Risk logic
    # --------------------

    NON_DEP_THRESHOLDS = {
        "high": 0.75,
        "mid": 0.5,
        "low": 0.3,
    }

    DEP_THRESHOLDS = {
        "high": 0.9,
        "mid": 0.7,
        "mild": 0.5,
    }

    K_NON_DEP = {
        "low": 6,
        "mid": 4,
        "high": 0,
    }

    K_DEP = {
        "mild": 5,
        "mid": 3,
        "high": 0,
    }


    if use_risk and hasattr(outputs, "risk_logits"):
        risk_logits = outputs.risk_logits
        risk_probs = torch.sigmoid(risk_logits)

        result["risk_logits"] = risk_logits.cpu()
        result["risk_probs"] = risk_probs.cpu()

        if optimal_thresholds_path:
            with open(optimal_thresholds_path, "r") as f:
                result["risk_thresholds"] = json.load(f)
        
        risk_ngrams = None
        if risk_ngrams_path:
            with open(risk_ngrams_path, "r", encoding="utf-8") as f:
                risk_ngrams = json.load(f)

        all_risk_reasoning = []
        all_risk_tiers = []
        all_risk_grounding = []

        for text_idx, text in enumerate(texts):
            risk_reasoning = {}
            risk_tiers = {}
            risk_grounding = {}

            for i, risk_name in enumerate(["depression", "selfharm", "suicidal", "grief"]):
                p_r = float(risk_probs[text_idx, i].item())
                tau_opt = result.get("risk_thresholds", {}).get(risk_name, 0.5)
                c_r = count_ngrams(text, risk_ngrams[risk_name]) if risk_ngrams else 0

                reasoning = False
                tier = "none"

                # --------------------
                # Depression (special regime)
                # --------------------
                if risk_name == "depression":
                    if p_r < DEP_THRESHOLDS["mild"]:
                        reasoning = False # BELOW optimal threshold -> suppress reasoning
                        tier = "none"
                    elif p_r >= tau_opt: # or p_r >= DEP_THRESHOLDS["high"]
                        reasoning = True
                        tier = "high"
                    elif p_r >= DEP_THRESHOLDS["mid"]:
                        tier = "mid"
                        if c_r >= K_DEP["mid"]:
                            reasoning = True
                    elif p_r >= DEP_THRESHOLDS["mild"]:
                        tier = "mild"
                        if c_r >= K_DEP["mild"]:
                            reasoning = True

                # --------------------
                # Non-depression risks
                # --------------------
                else:
                    if p_r < tau_opt:
                        # BELOW optimal threshold → suppress reasoning
                        reasoning = False
                        tier = "none"
                    elif p_r >= NON_DEP_THRESHOLDS["high"]:
                        reasoning = True
                        tier = "high"
                    elif p_r >= NON_DEP_THRESHOLDS["mid"]:
                        tier = "mid"
                        if c_r >= K_NON_DEP["high"]:
                            reasoning = True
                    elif p_r >= NON_DEP_THRESHOLDS["low"]:
                        tier = "low"
                        if c_r >= K_NON_DEP["mid"]:
                            reasoning = True
                    elif p_r >= tau_opt:
                        tier = "low"
                        if c_r >= K_NON_DEP["low"]:
                            reasoning = True
                    if risk_name == "selfharm":
                        if reasoning and c_r == 0:
                            tier = "ambiguous"

                grounding = []

                if (risk_ngrams is not None):
                    grounding = extract_textual_grounding(
                        text=text,
                        risk_ngrams=risk_ngrams.get(risk_name, [])
                    )

                risk_reasoning[f"{risk_name}_reasoning"] = reasoning
                risk_tiers[f"{risk_name}_tier"] = tier
                risk_grounding[risk_name] = grounding

            all_risk_reasoning.append(risk_reasoning)
            all_risk_tiers.append(risk_tiers)
            all_risk_grounding.append(risk_grounding)

        result["risk_reasoning"] = all_risk_reasoning
        result["risk_tiers"] = all_risk_tiers
        result["risk_textual_grounding"] = all_risk_grounding



    # --------------------
    # Subtheme inference (post-hoc, optional)
    # --------------------
    if use_subthemes:
        if lexicon_path is None:
            raise ValueError("lexicon_path must be provided when use_subthemes=True")
        if emotion_names is None:
            raise ValueError("emotion_names must be provided when use_subthemes=True")
        """
        Important properties:
        - Lexicon never sees logits
        - Lexicon never affects predictions
        - Emotion probabilities act as soft priors
        - Keywords act as symbolic anchors

        This makes the system:
        - Hybrid
        - Explainable
        - Auditable
        """
        lexicon = ThemeLexicon(lexicon_path)
        inferencer = SubthemeInferencer(
            lexicon=lexicon,
            alpha=alpha,
            beta=beta
        )


        all_subtheme_results = []
        all_lexicon_risk_flags = []

        for i, text in enumerate(texts):
            probs = emotion_probs[i].detach().cpu().tolist()
            lexicon_risk_flags = {
                "depression": False,
                "selfharm": False,
                "suicidal": False,
                "grief": False
            }

            emotion_probs_dict = {
                emotion_names[j]: float(probs[j])
                for j in range(len(emotion_names))
            }

            subtheme_scores = inferencer.infer(
                text=text,
                emotion_probs=emotion_probs_dict,
                return_evidence=True
            )

            all_subtheme_results.append(subtheme_scores)

            for s in subtheme_scores:
                if s["score"] <= 0.0:
                    continue
                if s["score"] < 1.0:
                    continue
                name = s["subtheme"].lower()

                if name == "depression":
                    lexicon_risk_flags["depression"] = True
                elif name in {"self-harm risk", "self harm", "selfharm"}:
                    lexicon_risk_flags["selfharm"] = True
                elif name in {"suicidal", "suicidal ideation"}:
                    lexicon_risk_flags["suicidal"] = True
                elif name == "grief":
                    lexicon_risk_flags["grief"] = True
                
            all_lexicon_risk_flags.append(lexicon_risk_flags)


        result["subthemes"] = all_subtheme_results
        result["lexicon_risk_evidence"] = all_lexicon_risk_flags


    return result
