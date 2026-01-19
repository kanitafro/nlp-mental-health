# project-root/bert/lexicon_utils.py
import json
import re

MIN_EMOTION_CONF = 0.10
MAX_EMOTIONS_PER_SUBTHEME = 2

def format_evidence_for_display(evidence: dict) -> str:
    parts = []

    if evidence.get("ngrams"):
        ngrams = ", ".join(e["text"] for e in evidence["ngrams"])
        parts.append(f"lexical: {ngrams}")

    if evidence.get("emotions"):
        emotions = ", ".join(
            f"{e['label']} ({e['confidence']:.2f})"
            for e in evidence["emotions"]
        )
        parts.append(f"emotional: {emotions}")

    if not parts:
        return "evidence: none"

    return "evidence: " + " | ".join(parts)


class ThemeLexicon:
    """
    ThemeLexicon loads a JSON lexicon mapping themes to keywords
    and provides a method to convert a text into a lexicon feature vector.
    """
    
    def __init__(self, lexicon_path):
        with open(lexicon_path, "r", encoding="utf-8") as f:
            self.lexicon = json.load(f)
        
        # Flatten hierarchical lexicon into a list of subthemes
        self.subthemes = []
        for theme, subtheme_dict in self.lexicon.items():
            for subtheme, info in subtheme_dict.items():
                self.subthemes.append({
                    "theme": theme,
                    "subtheme": subtheme,
                    "keywords": [kw.lower() for kw in info.get("keywords", [])],
                    "emotions": info.get("emotions", []),
                    "requires_lexical_evidence": info.get("requires_lexical_evidence", False)
                })


        # Preprocess keywords: lowercase for matching
        #self.lexicon_lower = {theme: [kw.lower() for kw in kws] for theme, kws in self.lexicon.items()}


class SubthemeInferencer:
    def __init__(self, lexicon: ThemeLexicon, alpha=1.0, beta=1.0):
        self.lexicon = lexicon
        self.alpha = alpha  # weight for keyword matches
        self.beta = beta    # weight for emotion alignment
    def infer(self, text: str, emotion_probs: dict, return_evidence: bool = True):
        """
        text: raw string
        emotion_probs: dict of {emotion_name: probability}
        Returns: list of scored subthemes sorted by score descending
        """
        text_lower = text.lower()
        subtheme_scores = []
        subtheme_evidence = {}

        for s in self.lexicon.subthemes:
            evidence = {
                "emotions": [],
                "ngrams": []
            }

            # 1. Keyword signal (count-based)
            matched_keywords = []
            for kw in s["keywords"]:
                if len(kw) < 4:
                    continue  # hard floor: kills "er", "re", etc.

                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, text_lower):
                    matched_keywords.append(kw)

            keyword_hits = len(matched_keywords)
            keyword_match = min(keyword_hits / 1.5, 1.0)

            if keyword_hits > 0:
                for kw in matched_keywords:
                    evidence["ngrams"].append({
                        "text": kw,
                        "contribution": self.alpha * 1.0
                    })

            # 2. Lexical eligibility gate
            if s["requires_lexical_evidence"] and keyword_hits == 0:
                score = 0.0
                if return_evidence:
                    evidence["blocked"] = "requires_lexical_evidence"
            else:
                # 3. Emotion signal
                emotion_score = 0.0
                emotion_contributions = []

                for e in s["emotions"]:
                    p = emotion_probs.get(e, 0.0)
                    if p >= MIN_EMOTION_CONF:
                        emotion_contributions.append((e, p))

                emotion_contributions.sort(key=lambda x: x[1], reverse=True)
                emotion_contributions = emotion_contributions[:MAX_EMOTIONS_PER_SUBTHEME]

                emotion_score = 0.0
                for e, p in emotion_contributions:
                    emotion_score += p
                    evidence["emotions"].append({
                        "label": e,
                        "confidence": p,
                        "contribution": self.beta * p
                    })

                score = self.alpha * keyword_match + self.beta * emotion_score

            if score > 0.0:
                entry = {
                    "theme": s["theme"],
                    "subtheme": s["subtheme"],
                    "score": score
                }
                if return_evidence:
                    entry["evidence"] = evidence
                subtheme_scores.append(entry)


        return sorted(subtheme_scores, key=lambda x: x["score"], reverse=True)
