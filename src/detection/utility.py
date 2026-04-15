from __future__ import annotations


KEY_TERMS = {
    "battery", "charge", "energy", "consumption", "range",
    "speed", "acceleration", "efficiency", "vehicle", "driving",
}


def utility_score(output_text: str) -> float:
    text = output_text.lower()
    term_hits = sum(1 for term in KEY_TERMS if term in text)
    brevity_penalty = 0.0 if len(text.split()) <= 80 else 0.2
    empty_penalty = 1.0 if not text.strip() else 0.0

    score = min(term_hits / max(len(KEY_TERMS) * 0.35, 1), 1.0)
    score = max(score - brevity_penalty - empty_penalty, 0.0)

    return round(score, 4)