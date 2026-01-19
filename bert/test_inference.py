# project-root/bert/test_inference.py
from collections import defaultdict
import json
from os import makedirs
import torch
from typing import Dict, List

from lexicon_utils import format_evidence_for_display
from inference import run_inference
from multitask_model import BertEmotionRiskModel

from lenient_decoding import batch_lenient_decode

EMOTION_NAMES = ["anger", "fear", "joy", "love", "sadness", "surprise"]
RISK_NAMES = ["depression", "selfharm", "suicidal", "grief"]


def explain_results(
    results: Dict,
    emotion_names: List[str] = None,
    risk_names: List[str] = None,
    max_subthemes: int = 5,
    subtheme_score_threshold: float = 0.0,
    delta: float = 0.05,
    min_emotion_confidence: float = 0.25,
) -> str:
    """
    Converts structured inference results into a human-readable explanation.

    Args:
        results: output dict from run_inference
        emotion_names: ordered list matching emotion logits
        risk_names: ordered list matching risk logits
        max_subthemes: maximum subthemes to mention
        subtheme_score_threshold: minimum score for subtheme inclusion
        delta: max difference for mixed-emotion acceptance
        min_emotion_confidence: min prob for emotion acceptance

    Returns:
        Natural language explanation string
    """
    lines = []
    
    # ----------------------------
    # 1. Emotion interpretation (uncertainty-aware)
    # ----------------------------
    if emotion_names is not None:
        probs = results["emotion_probs"][0].tolist()
        emotion_pairs = list(zip(emotion_names, probs))
        emotion_pairs.sort(key=lambda x: x[1], reverse=True)

        top_name, top_prob = emotion_pairs[0]
        second_name, second_prob = emotion_pairs[1]

        accepted = set()

        # Case 1: no emotion confident enough
        if top_prob < min_emotion_confidence:
            accepted = set()

        # Case 2: mixed affect (delta window)
        elif (top_prob - second_prob) < delta:
            accepted = {
                name for name, prob in emotion_pairs
                if prob >= min_emotion_confidence and (top_prob - prob) < delta
            }

        # Case 3: clear dominant emotion
        else:
            accepted = {top_name}

        id2label = {i: name for i, name in enumerate(emotion_names)}

        lenient_sets = batch_lenient_decode(
            results["emotion_probs"],
            id2label=id2label,
            min_confidence=min_emotion_confidence
        )

        lenient_accepted = lenient_sets[0]

        if not accepted:
            lines.append(
                "The text expresses mixed and restrained emotions without a single dominant affective signal."
            )
        elif len(accepted) == 1:
            lines.append(
                f"The dominant emotion expressed in the text is {next(iter(accepted))}."
            )
        else:
            ordered = [
                e for e, _ in emotion_pairs
                if e in accepted
            ]
            lines.append(
                "The emotional expression reflects a combination of "
                + ", ".join(ordered[:-1])
                + " and "
                + ordered[-1]
                + "."
            )

        """top_name, top_prob = emotion_pairs[0]
        second_name, second_prob = emotion_pairs[1]

        # Case 1: No emotion sufficiently confident
        if top_prob < MIN_CONF:
            lines.append(
                "The emotional tone of the text is weakly expressed and does not show a clear affective signal."
            )

        # Case 2: Mixed affect (near-tie at the top)
        elif (top_prob - second_prob) < DELTA:
            mixed = [
                name for name, prob in emotion_pairs
                if prob >= MIN_CONF and (top_prob - prob) < DELTA
            ]
            mixed_str = ""
            for i, name in enumerate(mixed):
                mixed_str += name
                if i < len(mixed) - 2:
                    mixed_str += ", "
                elif i == len(mixed) - 2:
                    mixed_str += ", and "
            #mixed_str = ", ".join(sorted(mixed))
            lines.append(
                f"The emotional expression is mixed, with overlapping signals of {mixed_str}, "
                f"and no clearly dominant emotion."
            )

        # Case 3: Dominant emotion
        else:
            lines.append(
                f"The dominant emotion expressed in the text is {top_name}."
            )"""

    # ----------------------------
    # 2. Theme / subtheme summary
    # ----------------------------
    if "subthemes" in results:
        subtheme_results = results["subthemes"][0]

        # Suppress explicit mental-health subthemes if any risk reasoning is active
        suppress_risk_subthemes = False
        if "risk_reasoning" in results:
            suppress_risk_subthemes = any(
                results["risk_reasoning"].get(f"{r}_reasoning", False)
                for r in ["depression", "selfharm", "suicidal"]
            )

        RISK_SUBTHEMES = {
            "depression",
            "self-harm risk",
            "suicidal"
        }

        # Filter by score threshold
        filtered = []
        added_MH_theme = False
        for s in subtheme_results:
            if s["score"] <= subtheme_score_threshold:
                continue

            if "evidence" in s:
                ev = s["evidence"]
                if "emotions" in ev:
                    ev_emotions = ev["emotions"]
                    if isinstance(ev_emotions, str):
                        ev_emotions = {ev_emotions}
                    elif isinstance(ev_emotions, list):
                        # Extract labels when evidence entries are dicts with label/confidence
                        extracted = []
                        for emo in ev_emotions:
                            if isinstance(emo, dict) and "label" in emo:
                                extracted.append(emo["label"])
                            elif isinstance(emo, str):
                                extracted.append(emo)
                        ev_emotions = set(extracted)
                    else:
                        ev_emotions = set()

                    if not ev_emotions.intersection(lenient_accepted):
                        continue

            name = s["subtheme"].lower()

            if suppress_risk_subthemes and name in RISK_SUBTHEMES:
                if not added_MH_theme:
                    filtered.append({
                        "theme": "Mental Health",
                        "subtheme": "various mental health indicators",
                        "score": s["score"]
                    })
                    added_MH_theme = True
                continue

            filtered.append(s)

        if filtered:
            themes = defaultdict(list)

            for s in filtered:
                themes[s["theme"]].append(s)

            for theme, subs in themes.items():
                subs = sorted(subs, key=lambda x: x["score"], reverse=True)
                sub_names = [s["subtheme"].lower() for s in subs[:max_subthemes]]

                if sub_names and (not added_MH_theme or theme != "Mental Health"):
                    lines.append(
                        f"The text discusses {theme}, particularly {', '.join(sub_names)}."
                    )
                elif sub_names and added_MH_theme and theme == "Mental Health":
                    lines.append(
                        f"The text discusses {', '.join(sub_names)}."
                    )
                else:
                    lines.append(
                        f"The text discusses {theme}."
                    )

    # ---------------------------- 
    # 3. Risk interpretation (thresholded) 
    # ----------------------------

    if risk_names is not None and "risk_probs" in results:
        risk_probs = results["risk_probs"][0].tolist()
        thresholds = results.get("risk_thresholds", {})

        if risk_names is not None and "risk_reasoning" in results:
            interpreted = []

            for risk in risk_names:
                p_r = results["risk_probs"][0][RISK_NAMES.index(risk)]
                #tau_opt = thresholds[risk]  # should be loaded from your thresholds dict
                tau_opt = thresholds.get(risk)
                if tau_opt is None:
                    continue
                
                if results["risk_reasoning"].get(f"{risk}_reasoning", False):
                    tier = results["risk_tiers"].get(f"{risk}_tier", "none")

                    if p_r < tau_opt and risk != "depression":
                        continue 

                    if risk == "depression":
                        if tier == "high":
                            if results["lexicon_risk_evidence"][0].get("depression", False):
                                interpreted.append("strong indicators of persistent depressive patterns")
                        elif tier == "mid":
                            if results["lexicon_risk_evidence"][0].get("depression", False):
                                interpreted.append("emerging depressive signals suggesting sustained emotional distress")
                        elif tier == "mild":
                            if results["lexicon_risk_evidence"][0].get("depression", False):
                                interpreted.append("mild depressive indicators")

                    elif risk == "selfharm":
                        if tier in {"mid", "low"}:
                            if results["lexicon_risk_evidence"][0].get("selfharm", False):
                                interpreted.append("developing self-harm related signals")
                        elif tier == "ambiguous":
                            interpreted.append("elevated self-harm related signals driven primarily by emotional patterns rather than explicit self-harm language")
                        elif tier == "high":
                            interpreted.append("elevated self-harm related signals")

                    elif risk == "suicidal":
                        if tier in {"mid", "low"}:
                            if results["lexicon_risk_evidence"][0].get("suicidal", False):
                                interpreted.append("developing suicidal risk indicators")
                        elif tier == "high":
                            interpreted.append("elevated suicidal risk indicators")

                    elif risk == "grief":
                        if tier in {"mid", "low"}:
                            if results["lexicon_risk_evidence"][0].get("grief", False):
                                interpreted.append("signals somewhat consistent with grief-related emotional processing")
                        elif tier == "high":
                            interpreted.append("signals consistent with grief-related emotional processing")

            if interpreted:
                lines.append(
                    "Additionally, the analysis detected " + "; ".join(interpreted) + "."
                )

    return " ".join(lines)

# ====================================================
# JSON RECORD BUILDER
# ====================================================
def build_json_record(text: str, results: Dict, index: int) -> Dict:
    """
    Builds a JSON-safe record for a single text.
    This is NEVER used for console output.
    """
    record = {
        "text_id": index,
        "content": text,
        "emotions": {},
        "subthemes": [],
        "risks": {}
    }

    # Emotions
    emotion_probs = results["emotion_probs"][index].tolist()
    record["emotions"] = {
        name: float(prob)
        for name, prob in zip(EMOTION_NAMES, emotion_probs)
    }

    # Subthemes (FULL, raw — includes evidence dicts)
    if "subthemes" in results:
        for s in results["subthemes"][index]:
            if s.get("score", 0.0) > 0.5: # only include subthemes with score > 0.5 to not overpopulate json
                record["subthemes"].append(s)

    # Risks
    if "risk_probs" in results:
        record["risks"]["probabilities"] = {
            name: float(prob)
            for name, prob in zip(
                RISK_NAMES,
                results["risk_probs"][index].tolist()
            )
        }

    if "risk_preds" in results:
        record["risks"]["decisions"] = {
            name: bool(pred)
            for name, pred in zip(
                RISK_NAMES,
                results["risk_preds"][index].tolist()
            )
        }

    if "risk_tiers" in results:
        record["risks"]["tiers"] = results["risk_tiers"][index]

    if "risk_reasoning" in results:
        record["risks"]["reasoning"] = results["risk_reasoning"][index]
    
    if "risk_textual_grounding" in results:
        record["risks"]["textual_grounding"] = (
            results["risk_textual_grounding"][index]
        )


    return record


# =====================================================
#            MAIN INFERENCE WRAPPED LOGIC
# =====================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilbert-base-uncased"

    model = BertEmotionRiskModel(
        model_name=model_name,            # or whatever you trained with
        num_labels=6,                     # number of emotion classes
        use_risk=True,                    # YOU trained with risk flags
        dropout_rate=0.2                  # MUST match training
    )

    checkpoint_path = "checkpoints_v1_1/v1_1_epoch_2.pt"
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    risk_ngrams_path="../data/risk_labels/risk_ngrams_reviewed.json"

    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    texts=["It's been a while since I scrolled through Instagram and today I came for a scroll. "
        "Every single reel was about mental health. A lot of them are about healing your "
        "inner child, about forgiving yourself. I cried a bit to those because I know I don't "
        "deserve it. Not yet, I hope one day, though. But there were a couple about what I would "
        "change if I could go back in time. The point of those videos are to let go of regret, "
        "that life is fine as it is. That we grow by learning from mistakes. I usually agree "
        "but…I wish I had asked you something in those months I wasn't active. I wish I had "
        "said something. A chit chat at least.",
        "I'm riding in the car right now with my dad. We're on our way to our weekly grocery "
        "shopping. I turned to this document because I saw a person on the sidewalk stopping "
        "to take pictures of the blossomed trees. Made me smile wide. I love seeing people "
        "find beauty in nature, I love seeing people appreciate it and not being afraid to "
        "stop and appreciate it. This is something I had to learn over the past couple of "
        "years or so. Slow down, there's no shame in stopping to appreciate life. My camera "
        "roll is full of close-ups of blossoms, colorful weeds, and many kinds of flowers - "
        "those that I can recognize and those that I'm not able to name in a million years "
        "but I appreciate them the same nevertheless. I love these rare moments in which "
        "the world feels like a nice place when most of the time it's evil beyond comprehension.",
        "I’ve hid it well, but today I’ve been in bits. I know the age I’m at, this could be my "
        "last chance to go to the World Cup. I couldn’t get my mate Diogo Jota out of my head "
        "today. We spoke so much about the World Cup. He missed out last time because of injury, "
        "I missed out because Scotland didn’t qualify and we always discussed what it would be "
        "like going to the World Cup. I was in a bit of trouble in my room earlier. I think I hid "
        "it well from the boys. I know he’ll be somewhere smiling over me tonight. I couldn’t get "
        "him out of my head all day.",
        "To my boyfriend. You're my favorite person and I love you like I never loved anyone. We had an "
        "argument earlier that I can't even remember because I've drunk so much my memories have faded, "
        "but my feelings still haven't. All I remember is that you were annoyed at me and it hurts, it "
        "fucking hurts. I want to hurt myself too because I so so deserve it. So I did. I cut myself a lot." 
        "All I ever wanted.. I just wanted to feel loved. I understand that maybe the friends you play with "
        "are more interesting. I understand that you want to have a life of your own. I understand that your" 
        "family is more important. I understand that you are busy and you're not willing to waste your time with me. "
        "I understand that I've made you angry and brought you to be rude to me and explain it to me in the roughest "
        "way. But if only you knew how much it hurt me. I know I'm not enough. I know I'm insecure. I know I'm selfish. "
        "I know I'm easily jealous. I know you won't drop your life for me because I'm not worth it. I know you have "
        "goals and ambitions while I rot in my bed all day waiting for a reply. I know I probably won't ever get better. "
        "I wish you just dumped me. You don't imagine how hard I am crying right now and how much I want to slit my wrists "
        "open. You've just convinced me I'll never be enough and I have a lot of flaws. I was aware of it, but now you "
        "showed me it's true. I know I'm in constant need of reassurance and it pisses you off because you feel like I "
        "don't trust you. I know it pisses you off because convincing me that I am good enough is pointless to you, because "
        "I \"don't believe in you anyway\"(I don't believe in myself). I know you feel like you should stop trying to help me " 
        "because you find it's useless since you say your words don't reach me (trust me, I really want your help more than "
        "anything and I'm trying really hard to keep myself together and believe what you tell me). I know you're more thick "
        "skinned than me and I am just a coward (sensitive, cries easily). I wish I could be different. I wish I wouldn't "
        "overreact. I wish I didn't want you to be with me all the time because alone I feel like I'm a no one. I have nothingness "
        "inside of me. I'm just an empty void with the face of a human. It hurts me like I'm burning in hell when I get any kind of "
        "rejection from you. It hurts, it hurts it hurts. Please don't push me away just because I want to be close to you. Please "
        "don't get mad at me for being emotional and irrational. I just want to feel safe beside you. I just want you to offer me "
        "that protection I never had as a child. Deep down I'm still as sensitive as a child, I never grew up. I can't deal with "
        "feeling abandoned by you and I'm sorry. All I want is you to be my safe space and support me. There's nothing more I want. "
        "I don't want expensive gifts, I don't want expensive trips, I don't want money, I just want to be loved.",
        "For months after your death I've looked at life in such awe, looked at every detail thoroughly, so very mesmerised "
        "with its beauty. The sky, the chirps of the birds, the vibrancy of the colours, the grey when it rains. So I thought to "
        "myself that this all made me more alive than ever, even though I knew of the pain inside me that I could live with that "
        "guilt. I don't know how this hope turned into the darkest thoughts imaginable.",
        "I feel so lost. I don't know who I am anymore. I don't know what to do with my life. I feel like I'm just existing, not living. "
        "I don't want to die, but I don't want to live like this either. Every day is a struggle. I feel like I'm falling apart."
    ]
    
    results = run_inference(
        model=model,
        texts=texts,
        tokenizer_name=model_name,
        device=device,
        use_risk=True,
        use_subthemes=True,
        lexicon_path="../data/lexicon/lexicon_clean_6.json",
        emotion_names=EMOTION_NAMES,
        optimal_thresholds_path="saved_models/trained_model_v1_1/metrics/threshold_sweeps/optimal_thresholds.json",
        risk_ngrams_path=risk_ngrams_path
    )

    # =====================================================
    # JSON AGGREGATION (machine-readable, full evidence)
    # =====================================================
    json_output = {
        "model": model_name,
        "num_texts": len(texts),
        "results": []
    }

    for i, text in enumerate(texts):
        json_output["results"].append(
            build_json_record(text, results, i)
        )

    makedirs("json_files", exist_ok=True)
    with open("json_files/test_inference_output.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    for i, text in enumerate(texts):
        print(f"\n======================= TEXT {i+1} =======================\n")
        print(text)

        # Emotions
        print("\n=== EMOTION PROBABILITIES ===")
        emotion_probs = results["emotion_probs"][i].tolist()
        for name, prob in sorted(
            zip(EMOTION_NAMES, emotion_probs),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"{name:<10}: {prob:.3f}")

        # Subthemes
        if "subthemes" in results:
            print("\n=== SUBTHEME SCORES ===")
            for s in results["subthemes"][i][:10]:
                print(
                    f"{s['theme']:<15} → "
                    f"{s['subtheme']:<25} "
                    f"(score={s['score']:.3f})"
                )
                if "evidence" in s:
                    print("  " + format_evidence_for_display(s["evidence"]))

        # Risks
        if "risk_probs" in results:
            print("\n=== RISK PROBABILITIES ===")
            for name, prob in zip(
                RISK_NAMES,
                results["risk_probs"][i].tolist()
            ):
                print(f"{name:<10}: {prob:.3f}")

        # Explanation
        print("\n=== INTERPRETATION ===")
        # Build single-text result dict for explain_results
        single_result = {
            **results,
            "emotion_probs": results["emotion_probs"][i:i+1],
            "subthemes": [results["subthemes"][i]],
            "lexicon_risk_evidence": [results["lexicon_risk_evidence"][i]]
        }
        
        # Add risk reasoning/tiers if they exist
        if "risk_reasoning" in results:
            single_result["risk_reasoning"] = results["risk_reasoning"][i]
        if "risk_tiers" in results:
            single_result["risk_tiers"] = results["risk_tiers"][i]
        
        explanation = explain_results(
            single_result,
            emotion_names=EMOTION_NAMES,
            risk_names=RISK_NAMES,
            subtheme_score_threshold=1.0,
        )
        
        print(explanation)



if __name__ == "__main__":
    main()


 