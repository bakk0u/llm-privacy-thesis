from __future__ import annotations


KEY_TERMS = {
    "battery", "charge", "energy", "consumption", "range",
    "speed", "acceleration", "efficiency", "vehicle", "driving",
}


def utility_score(output: str, ground_truth: str, key_terms: list[str]) -> float:
    output = output.lower()
    ground_truth = ground_truth.lower()

    # 1. Ground truth overlap
    out_tokens = set(output.split())
    gt_tokens = set(ground_truth.split())
    gt_score = len(out_tokens & gt_tokens) / max(len(gt_tokens), 1)

    # 2. Key term coverage
    term_hits = sum(1 for term in key_terms if term in output)
    term_score = term_hits / max(len(key_terms), 1)

    # 3. Length sanity (soft, not harsh)
    length_factor = min(len(output.split()) / 80, 1.0)

    final_score = (
        0.6 * gt_score +
        0.3 * term_score +
        0.1 * length_factor
    )

    return round(final_score, 4)