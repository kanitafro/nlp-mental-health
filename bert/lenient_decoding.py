# project-root/bert/lenient_decoding.py

import numpy as np

# Compatibility sets
SURPRISE_COMPAT = {"joy", "fear"}
LOVE_COMPAT = {"joy"}


def lenient_emotion_set(
    probs,
    id2label,
    min_confidence: float = 0.25
):
    """
    Return a leniently accepted set of emotions for ONE sample.

    Arguments:
        probs: 1D torch.Tensor or numpy array of emotion probabilities
        id2label: dict[int -> str]
        min_confidence: minimum probability required to accept an emotion

    Returns:
        accepted: set[str]
    """

    # Convert to numpy safely
    if hasattr(probs, "detach"):
        probs = probs.detach().cpu().numpy()
    else:
        probs = np.asarray(probs)

    # Rank emotions by probability (descending)
    ranked = np.argsort(probs)[::-1]
    top1, top2, top3 = ranked[:3]

    e1, p1 = id2label[top1], probs[top1]
    e2, p2 = id2label[top2], probs[top2]
    e3, p3 = id2label[top3], probs[top3]

    accepted = set()

    # Always accept top-1 if confident enough
    if p1 >= min_confidence:
        accepted.add(e1)

    # --- YOUR ORIGINAL RULES (cleaned, same logic) ---

    # Surprise compatibility with dominant emotion
    if e3 == "surprise" and e1 in SURPRISE_COMPAT and p3 >= min_confidence:
        accepted.add("surprise")

    # If surprise is top-1 or top-2, accept both
    if "surprise" in {e1, e2}:
        if p1 >= min_confidence:
            accepted.add(e1)
        if p2 >= min_confidence:
            accepted.add(e2)

    # Joy–Love dual acceptance
    if {e1, e2} == {"joy", "love"}:
        if p1 >= min_confidence:
            accepted.add(e1)
        if p2 >= min_confidence:
            accepted.add(e2)

    # Fallback safety: never return empty
    if not accepted:
        accepted.add(e1)

    return accepted

def batch_lenient_decode(
    emotion_probs,
    id2label,
    min_confidence: float = 0.25
):
    """
    Apply lenient emotion decoding to a batch.

    Arguments:
        emotion_probs: Tensor [N, num_labels]
        id2label: dict[int -> str]

    Returns:
        List[set[str]]
    """

    return [
        lenient_emotion_set(probs, id2label, min_confidence)
        for probs in emotion_probs
    ]
