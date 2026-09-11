# demo/app.py
"""
Streamlit app for journal-entry emotion analysis using a fine-tuned
multi-label DistilBERT model (28 GoEmotions).

Features
--------
- Sentence-level segmentation (nltk with regex fallback)
- Sliding-window segmentation (configurable window size & stride)
- Adjustable decision threshold for sigmoid outputs
- Per-segment emotion predictions
- Aggregated emotion profile for the whole entry
- Interactive timeline, heatmap, and per-segment tables
- CSV export of per-segment results
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bert.xai.explain_multilabel import (
    LABELS,
    load_model,
    EmotionPredictor,
)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

CHECKPOINT = (
    PROJECT_ROOT
    / "bert"
    / "saved_models"
    / "finetuned_model_v2_7_1_2_3"
    / "best_model.pt"
)

POSITIVE_EMOTIONS = {
    "admiration", "caring", "gratitude", "love", "desire",
    "amusement", "excitement", "joy", "optimism", "pride", "relief",
}
NEGATIVE_EMOTIONS = {
    "anger", "annoyance", "disapproval", "disgust", "disappointment",
    "embarrassment", "remorse", "grief", "sadness", "fear", "nervousness",
}
AMBIGUOUS_EMOTIONS = {
    "confusion", "curiosity", "realization", "excitement", "surprise",
}

POLARITY_COLORS = {
    "positive": "#FF5B9A",
    "negative": "#09ABD1",
    "neutral": "#B1E692",
    "ambiguous": "#F89E6A",
}


def polarity_of(emotion: str) -> str:
    if emotion in POSITIVE_EMOTIONS:
        return "positive"
    if emotion in NEGATIVE_EMOTIONS:
        return "negative"
    if emotion == "neutral":
        return "neutral"
    return "ambiguous"


# ------------------------------------------------------------
# Negation detection
# ------------------------------------------------------------

NEGATION_TOKENS = {
    "not", "n't", "nt", "dont", "don't", "doesnt", "doesn't",
    "wont", "won't", "cant", "can't", "cannot", "couldnt", "couldn't",
    "never", "no", "none", "nobody", "nothing", "neither", "nor",
    "hardly", "barely", "scarcely", "without",
}

# Lexical cues per emotion. These are the words that, when preceded or
# followed by a negation token, are likely to have their polarity flipped.
EMOTION_CUES = {
    "fear":        {"terrified", "scared", "afraid", "fear", "worried", "worry"},
    "nervousness": {"nervous", "anxious", "anxiety", "worried", "worry"},
    "surprise":    {"surprised", "shocked", "shocking", "surprise", "unexpected"},
    "approval":    {"agree", "right", "fine", "good", "okay", "ok", "yes"},
    "disapproval": {"wrong", "bad", "shouldn't", "shouldnt", "forgive"},
    "joy":         {"happy", "glad", "enjoy", "love", "excited"},
    "sadness":     {"sad", "miss", "hurt", "cry", "unhappy"},
    "anger":       {"angry", "mad", "furious"},
    "caring":      {"worry", "care", "support", "help"},
    "love":        {"love", "adore", "dearly"},
    "optimism":    {"hope", "hopeful", "wish", "optimistic"},
    "desire":      {"want", "wish", "hope", "long"},
    "remorse":     {"sorry", "apologize", "apologise"},
    "gratitude":   {"thank", "thanks", "grateful", "appreciate"},
}


def has_negation_near_emotion(text, emotion, window=2):
    """
    Return True if a negation token appears within `window` words of any
    lexical cue for the given emotion. This is a heuristic, not a
    linguistic parser; it is meant to flag likely-flipped predictions
    for human review.
    """
    cues = EMOTION_CUES.get(emotion)
    if not cues:
        return False

    words = [w.lower().strip(".,!?;:()\"'“”") for w in text.split()]

    for i, w in enumerate(words):
        if w in cues:
            lo = max(0, i - window)
            hi = min(len(words), i + window + 1)
            if any(n in NEGATION_TOKENS for n in words[lo:hi]):
                return True
    return False

# ------------------------------------------------------------
# Cached model loading
# ------------------------------------------------------------

@st.cache_resource(show_spinner="Loading fine-tuned model…")
def get_predictor():
    if not CHECKPOINT.exists():
        st.error(f"Checkpoint not found: {CHECKPOINT}")
        st.stop()
    model, tokenizer, device = load_model(
        str(CHECKPOINT),
        num_labels=len(LABELS),
    )
    return EmotionPredictor(model, tokenizer, device, max_length=128)


# ------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------

def split_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        from nltk.tokenize import sent_tokenize

        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def sliding_window_chunks(text: str, window_size: int, stride: int):
    words = text.split()
    if not words:
        return []
    stride = max(1, int(stride))
    window_size = max(1, int(window_size))
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + window_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if i + window_size >= len(words):
            break
        i += stride
    return chunks


def build_segments(text, strategy, window_size=10, stride=5):
    if strategy == "Sentence-level":
        return split_sentences(text)
    if strategy == "Sliding window":
        return sliding_window_chunks(text, window_size, stride)
    if strategy == "Whole entry":
        return [text.strip()] if text and text.strip() else []
    raise ValueError(f"Unknown strategy: {strategy}")


# ------------------------------------------------------------
# Inference + aggregation
# ------------------------------------------------------------

def run_inference(
    predictor,
    segments,
    threshold,
    neutral_strategy="Same as others",
    neutral_threshold=0.75,
    neutral_penalty=0.15,
    fallback_threshold=0.25,
    max_emotions=None,
    relative_margin=None,
):
    """
    Run multi-label inference with optional neutral recalibration.

    neutral_strategy:
        - "Same as others"                : neutral uses the same threshold
        - "Higher threshold for neutral"  : neutral needs `neutral_threshold`
        - "Penalize neutral probability"  : subtract `neutral_penalty` from neutral prob
        - "Drop neutral entirely"         : never predict neutral
        - "Fallback if only neutral"      : if only neutral crosses the threshold,
                                            take the top non-neutral emotion above
                                            `fallback_threshold` instead

    max_emotions:
        If not None, keep at most this many emotions per segment (top-K by prob).

    relative_margin:
        If not None, after thresholding, only keep emotions whose probability is
        within `relative_margin` of the top prediction for that segment.
        Set to None to disable.
    """
    if not segments:
        return [], np.zeros((0, len(LABELS)), dtype=np.float32)

    probs = predictor(segments)  # [n_segments, 28]
    neutral_idx = LABELS.index("neutral")

    adjusted = probs.copy()

    if neutral_strategy == "Penalize neutral probability":
        adjusted[:, neutral_idx] = np.clip(
            adjusted[:, neutral_idx] - neutral_penalty, 0.0, 1.0
        )
    elif neutral_strategy == "Drop neutral entirely":
        adjusted[:, neutral_idx] = 0.0

    per_segment = []
    for i, seg in enumerate(segments):
        row = adjusted[i]

        effective_threshold = threshold
        if neutral_strategy == "Higher threshold for neutral":
            candidates = []
            for j in range(len(LABELS)):
                if LABELS[j] == "neutral":
                    if probs[i, j] >= neutral_threshold:
                        candidates.append((j, probs[i, j]))
                else:
                    if probs[i, j] >= threshold:
                        candidates.append((j, probs[i, j]))
            candidates.sort(key=lambda x: -x[1])
        else:
            candidates = [
                (j, float(row[j]))
                for j in range(len(LABELS))
                if row[j] >= effective_threshold
            ]
            candidates.sort(key=lambda x: -x[1])

        # --- Fallback: if only neutral is present, look for the next non-neutral ---
        if neutral_strategy == "Fallback if only neutral":
            non_neutral = [c for c in candidates if LABELS[c[0]] != "neutral"]
            if not non_neutral:
                best = max(
                    ((j, probs[i, j]) for j in range(len(LABELS))
                     if LABELS[j] != "neutral"),
                    key=lambda x: x[1],
                )
                if best[1] >= fallback_threshold:
                    candidates = [best]

        # --- Optional: relative margin filter ---
        if relative_margin is not None and candidates:
            top_prob = candidates[0][1]
            candidates = [
                c for c in candidates if top_prob - c[1] <= relative_margin
            ]

        # --- Optional: top-K cap ---
        if max_emotions is not None and len(candidates) > max_emotions:
            candidates = candidates[:max_emotions]

        preds = []
        for j, p in candidates:
            emotion = LABELS[j]
            preds.append({
                "emotion": emotion,
                "prob": float(p),
                "negation_warning": has_negation_near_emotion(seg, emotion),
            })

        per_segment.append({
            "segment_index": i + 1,
            "text": seg,
            "word_count": len(seg.split()),
            "predictions": preds,
            "neutral_prob": float(probs[i, neutral_idx]),
        })

    return per_segment, probs


def aggregate_scores(per_segment, probs, method):
    """Return dict {emotion: score} according to the chosen aggregation method."""
    if probs.size == 0:
        return {}

    if method == "Count of segments above threshold":
        counts = {label: 0 for label in LABELS}
        for seg in per_segment:
            for p in seg["predictions"]:
                counts[p["emotion"]] += 1
        return counts

    if method == "Sum of probabilities":
        return {LABELS[j]: float(probs[:, j].sum()) for j in range(len(LABELS))}

    if method == "Mean of probabilities":
        return {LABELS[j]: float(probs[:, j].mean()) for j in range(len(LABELS))}

    if method == "Max probability":
        return {LABELS[j]: float(probs[:, j].max()) for j in range(len(LABELS))}

    raise ValueError(method)


# ------------------------------------------------------------
# Visualisations
# ------------------------------------------------------------

def timeline_figure(probs, threshold, max_emotions=None):
    n_segments = probs.shape[0]
    if n_segments == 0:
        return go.Figure()

    active_mask = (probs >= threshold).any(axis=0)
    active_idx = [j for j in range(len(LABELS)) if active_mask[j]]

    if not active_idx:
        top = np.argsort(-probs.max(axis=0))[:5]
        active_idx = list(top)

    if max_emotions is not None and len(active_idx) > max_emotions:
        active_idx = sorted(
            active_idx,
            key=lambda j: -probs[:, j].max(),
        )[:max_emotions]

    x = list(range(1, n_segments + 1))

    fig = go.Figure()
    for j in active_idx:
        emo = LABELS[j]
        fig.add_trace(go.Scatter(
            x=x,
            y=probs[:, j],
            mode="lines+markers",
            name=emo,
            line=dict(
                color=POLARITY_COLORS[polarity_of(emo)],
            ),
            hovertemplate=(
                f"<b>{emo}</b><br>"
                "Segment %{x}<br>"
                "Probability %{y:.3f}<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"threshold = {threshold:.2f}",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Segment index",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(l=30, r=20, t=30, b=20),
    )
    return fig


def heatmap_figure(probs, threshold):
    if probs.size == 0:
        return go.Figure()

    active_mask = (probs >= threshold).any(axis=0)
    active_idx = [j for j in range(len(LABELS)) if active_mask[j]]
    if not active_idx:
        active_idx = list(np.argsort(-probs.max(axis=0))[:8])

    sub = probs[:, active_idx]
    labels = [LABELS[j] for j in active_idx]

    fig = px.imshow(
        sub.T,
        aspect="auto",
        color_continuous_scale=[
            [0.0, "#FFFFFF"],
            [0.25, "#FFD769"],
            [0.5, "#FF7F7F"],
            [1.0, "#8B1A1A"],
        ],
        zmin=0.0,
        zmax=1.0,
        labels=dict(x="Segment", y="Emotion", color="Probability"),
        x=[str(i + 1) for i in range(probs.shape[0])],
        y=labels,
    )
    fig.update_layout(
        height=max(320, 22 * len(labels)),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def aggregated_bar_figure(agg_scores, top_k=15):
    if not agg_scores:
        return go.Figure()

    items = sorted(agg_scores.items(), key=lambda kv: -kv[1])
    items = [x for x in items if x[1] > 0][:top_k]
    if not items:
        return go.Figure()

    emos = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [POLARITY_COLORS[polarity_of(e)] for e in emos]

    fig = go.Figure(go.Bar(
        x=vals,
        y=emos,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Aggregated score",
        yaxis_title="",
        height=max(320, 26 * len(emos)),
        margin=dict(l=20, r=40, t=20, b=20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

st.set_page_config(
    page_title="Journal Emotion Analysis",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Journal Entry Emotion Analysis")
st.caption(
    "Fine-tuned multi-label DistilBERT · 28 GoEmotions categories · "
    "sentence- and window-level trajectory analysis"
)

with st.sidebar:
    st.header("⚙️ Analysis settings")

    strategy = st.radio(
        "Segmentation strategy",
        ["Sentence-level", "Sliding window", "Whole entry"],
        index=0,
        help=(
            "Sentence-level: split by sentence boundaries. "
            "Sliding window: overlapping word windows. "
            "Whole entry: single prediction for the full text."
        ),
    )

    if strategy == "Sliding window":
        window_size = st.slider("Window size (words)", 3, 50, 10, 1)
        stride = st.slider("Stride (words)", 1, 25, 5, 1)
    else:
        window_size = 10
        stride = 5

    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Sigmoid probability cutoff for each emotion.",
    )

    st.markdown("---")
    st.subheader("Neutral recalibration")
    st.caption(
        "GoEmotions is dominated by neutral (Reddit comments). Journal entries "
        "are introspective, so neutral over-fires. These controls let you "
        "recalibrate the neutral decision boundary without retraining."
    )

    neutral_strategy = st.selectbox(
        "Neutral handling strategy",
        [
            "Same as others",
            "Higher threshold for neutral",
            "Penalize neutral probability",
            "Drop neutral entirely",
            "Fallback if only neutral",
        ],
        index=4,
    )

    neutral_threshold = 0.75
    neutral_penalty = 0.15
    fallback_threshold = 0.25

    if neutral_strategy == "Higher threshold for neutral":
        neutral_threshold = st.slider(
            "Neutral threshold",
            min_value=0.50,
            max_value=0.99,
            value=0.75,
            step=0.05,
            help="Neutral must exceed this. Other emotions still use the general threshold above.",
        )

    if neutral_strategy == "Penalize neutral probability":
        neutral_penalty = st.slider(
            "Neutral probability penalty",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.05,
            help="Subtracted from the neutral probability before thresholding.",
        )

    if neutral_strategy == "Fallback if only neutral":
        fallback_threshold = st.slider(
            "Fallback threshold",
            min_value=0.05,
            max_value=0.6,
            value=0.25,
            step=0.05,
            help=(
                "If only neutral crosses the general threshold, the best "
                "non-neutral emotion is used if it is at least this high."
            ),
        )

    st.markdown("---")
    st.subheader("Per-segment limits")

    max_emotions_per_segment = st.slider(
        "Max emotions per segment",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Cap the number of labels kept per segment (top-K by probability).",
    )

    relative_margin = st.slider(
        "Relative margin filter",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.05,
        help=(
            "If > 0, only keep emotions whose probability is within this margin "
            "of the top prediction for the segment. Set to 0 to disable."
        ),
    )

    if relative_margin == 0.0:
        relative_margin = None

    st.markdown("---")
    st.subheader("Aggregation")
    agg_method = st.selectbox(
        "Score aggregation method",
        [
            "Count of segments above threshold",
            "Mean of probabilities",
            "Sum of probabilities",
            "Max probability",
        ],
        index=1,
    )

    st.markdown("---")
    st.subheader("Timeline")
    max_emotions_timeline = st.slider(
        "Max emotions on timeline",
        min_value=3,
        max_value=28,
        value=10,
        step=1,
    )

    st.markdown("---")
    st.markdown(
        "**Polarity legend**  \n"
        "🔴 Positive  \n"
        "🔵 Negative  \n"
        "🟢 Neutral  \n"
        "🟠 Ambiguous / surprise"
    )

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

default_text = (
    "I woke up feeling anxious about the meeting today. "
    "But then my friend called and we laughed for an hour, and I felt so grateful. "
    "Later I realised I had been overthinking everything. "
    "I'm still a bit sad about last week, but overall I'm hopeful."
)

journal_text = st.text_area(
    "✍️ Paste or type your journal entry",
    value=default_text,
    height=200,
)

col_a, col_b = st.columns([1, 5])
with col_a:
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col_b:
    st.write("")  # spacer

# ------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------

if analyze_clicked:
    if not journal_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        predictor = get_predictor()
        segments = build_segments(journal_text, strategy, window_size, stride)

        if not segments:
            st.warning("No segments were produced from the input.")
        else:
            with st.spinner(f"Running model on {len(segments)} segment(s)…"):
                per_segment, probs = run_inference(
                    predictor,
                    segments,
                    threshold,
                    neutral_strategy=neutral_strategy,
                    neutral_threshold=neutral_threshold,
                    neutral_penalty=neutral_penalty,
                    fallback_threshold=fallback_threshold,
                    max_emotions=max_emotions_per_segment,
                    relative_margin=relative_margin,
                )

            st.session_state["per_segment"] = per_segment
            st.session_state["probs"] = probs
            st.session_state["segments"] = segments
            st.session_state["strategy"] = strategy
            st.session_state["threshold"] = threshold
            st.session_state["agg_method"] = agg_method

# ------------------------------------------------------------
# Render results
# ------------------------------------------------------------

if "per_segment" in st.session_state:
    per_segment = st.session_state["per_segment"]
    probs = st.session_state["probs"]
    segments = st.session_state["segments"]
    strategy_used = st.session_state["strategy"]
    threshold_used = st.session_state["threshold"]
    agg_method_used = st.session_state["agg_method"]

    st.markdown("---")
    st.subheader(
        f"📊 Results · strategy = *{strategy_used}* · "
        f"threshold = *{threshold_used:.2f}* · "
        f"segments = *{len(segments)}*"
    )

    tab_timeline, tab_segments, tab_agg, tab_heat, tab_export = st.tabs(
        ["📈 Timeline", "📝 Per-segment", "🏆 Aggregated", "🌡️ Heatmap", "⬇️ Export"]
    )

    # --------------------------------------------------------
    # Timeline tab
    # --------------------------------------------------------
    with tab_timeline:
        st.plotly_chart(
            timeline_figure(
                probs,
                threshold_used,
                max_emotions=max_emotions_timeline,
            ),
            use_container_width=True,
        )
        st.caption(
            "Each line is one emotion. The dashed line marks the decision "
            "threshold. Emotions that never cross the threshold are hidden "
            "unless no emotion is active, in which case the top-5 most "
            "probable emotions are shown."
        )

    # --------------------------------------------------------
    # Per-segment tab
    # --------------------------------------------------------
    with tab_segments:
        if not per_segment:
            st.info("No segments to display.")
        else:
            for seg in per_segment:
                preds = seg["predictions"]
                header = (
                    f"**Segment {seg['segment_index']}** · "
                    f"{seg['word_count']} word(s)"
                )
                with st.expander(header, expanded=(len(per_segment) <= 8)):
                    st.write(f"*{seg['text']}*")
                    if not preds:
                        st.write("No emotion above threshold.")
                    else:
                        df = pd.DataFrame(preds)
                        df["polarity"] = df["emotion"].apply(polarity_of)
                        df["Note"] = df["negation_warning"].apply(
                            lambda flagged: "⚠ negation nearby" if flagged else ""
                        )
                        df = df[["emotion", "polarity", "prob", "Note"]]
                        df.columns = ["Emotion", "Polarity", "Probability", "Note"]
                        st.dataframe(
                            df.style.format({"Probability": "{:.3f}"}),
                            hide_index=True,
                            use_container_width=True,
                        )

    # --------------------------------------------------------
    # Aggregated tab
    # --------------------------------------------------------
    with tab_agg:
        agg_scores = aggregate_scores(
            per_segment, probs, agg_method_used
        )
        if not agg_scores or all(v == 0 for v in agg_scores.values()):
            st.info("No emotion scored above zero with the current settings.")
        else:
            st.plotly_chart(
                aggregated_bar_figure(agg_scores, top_k=15),
                use_container_width=True,
            )
            st.caption(f"Aggregation method: *{agg_method_used}*.")

            pol_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "ambiguous": 0.0}
            for emo, val in agg_scores.items():
                pol_scores[polarity_of(emo)] += val

            total = sum(pol_scores.values())
            if total > 0:
                pol_df = pd.DataFrame({
                    "Polarity": list(pol_scores.keys()),
                    "Score": list(pol_scores.values()),
                })
                pol_df["Share"] = pol_df["Score"] / total
                st.markdown("**Polarity distribution**")
                st.dataframe(
                    pol_df.style.format({"Score": "{:.2f}", "Share": "{:.1%}"}),
                    hide_index=True,
                    use_container_width=True,
                )

    # --------------------------------------------------------
    # Heatmap tab
    # --------------------------------------------------------
    with tab_heat:
        st.plotly_chart(
            heatmap_figure(probs, threshold_used),
            use_container_width=True,
        )
        st.caption(
            "Colour intensity encodes the sigmoid probability of each "
            "emotion at each segment. Only emotions that cross the "
            "threshold at least once are shown."
        )

    # --------------------------------------------------------
    # Export tab
    # --------------------------------------------------------
    with tab_export:
        rows = []
        for seg in per_segment:
            if not seg["predictions"]:
                rows.append({
                    "segment_index": seg["segment_index"],
                    "word_count": seg["word_count"],
                    "text": seg["text"],
                    "emotion": "",
                    "probability": np.nan,
                    "negation_warning": False,
                })
            else:
                for p in seg["predictions"]:
                    rows.append({
                        "segment_index": seg["segment_index"],
                        "word_count": seg["word_count"],
                        "text": seg["text"],
                        "emotion": p["emotion"],
                        "probability": p["prob"],
                        "negation_warning": p.get("negation_warning", False),
                    })

        export_df = pd.DataFrame(rows)
        st.dataframe(export_df, hide_index=True, use_container_width=True)

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download per-segment predictions (CSV)",
            data=csv_bytes,
            file_name="journal_emotion_predictions.csv",
            mime="text/csv",
        )

        prob_df = pd.DataFrame(
            probs,
            columns=LABELS,
        )
        prob_df.insert(0, "segment_index", range(1, len(prob_df) + 1))
        prob_df.insert(1, "text", segments)

        prob_csv = prob_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download full probability matrix (CSV)",
            data=prob_csv,
            file_name="journal_probability_matrix.csv",
            mime="text/csv",
        )

else:
    st.info(
        "Enter a journal entry above, choose your settings in the sidebar, "
        "then click **Analyze**."
    )