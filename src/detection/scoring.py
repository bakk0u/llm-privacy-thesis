from __future__ import annotations

from typing import Any


def compute_leakage_score(
    findings: dict[str, dict[str, Any]],
    field_weights: dict[str, float],
) -> float:
    score = 0.0

    for field, details in findings.items():
        weight = float(field_weights.get(field, 1.0))
        if details["exact"]:
            score += weight
        elif details["partial"]:
            score += 0.5 * weight

    return round(score, 4)


def compute_exposure_rate(
    findings: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> float:
    if not record:
        return 0.0
    return round(len(findings) / len(record), 4)