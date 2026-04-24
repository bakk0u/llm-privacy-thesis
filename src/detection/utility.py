from __future__ import annotations

from rouge_score import rouge_scorer

KEY_TERMS = {
    "battery", "charge", "energy", "consumption", "range",
    "speed", "acceleration", "efficiency", "vehicle", "driving",
}

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 between hypothesis and reference."""
    scores = _rouge.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def utility_score(output: str, ground_truth: str, key_terms: list[str]) -> float:
    """
    Composite utility score:
      60% ROUGE-L F1 against ground truth  (semantic overlap, not bag-of-words)
      30% domain key-term coverage          (task-relevant vocabulary)
      10% non-redundancy                    (penalises verbatim prompt echoing)

    Weights are fixed at 0.6/0.3/0.1 — document in thesis appendix if changed.
    """
    output_lower = output.lower()
    gt_lower = ground_truth.lower()

    # 1. ROUGE-L replaces bag-of-words token overlap
    rouge_score = _rouge_l(output_lower, gt_lower)

    # 2. Domain key-term coverage
    term_hits = sum(1 for term in key_terms if term in output_lower)
    term_score = term_hits / max(len(key_terms), 1)

    # 3. Non-redundancy: penalise if output is largely a copy of the ground truth.
    # High ROUGE-L + short unique content suggests echoing, not generation.
    output_tokens = set(output_lower.split())
    gt_tokens = set(gt_lower.split())
    unique_ratio = len(output_tokens - gt_tokens) / max(len(output_tokens), 1)
    non_redundancy = min(unique_ratio * 2, 1.0)  # scale: 0.5 unique → 1.0

    final_score = (
        0.6 * rouge_score +
        0.3 * term_score +
        0.1 * non_redundancy
    )

    return round(final_score, 4)