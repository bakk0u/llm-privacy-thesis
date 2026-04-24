from __future__ import annotations

from typing import Any

from src.detection.leakage_rules import SENSITIVE_FIELDS


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


def compute_normalized_leakage_score(
    findings: dict[str, dict[str, Any]],
    field_weights: dict[str, float],
) -> float:
    """Leakage score divided by maximum possible score across all sensitive fields."""
    raw = compute_leakage_score(findings, field_weights)
    max_possible = sum(float(field_weights.get(f, 1.0)) for f in SENSITIVE_FIELDS)
    if max_possible == 0:
        return 0.0
    return round(raw / max_possible, 4)


def compute_exposure_rate(
    findings: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> float:
    """Fraction of sensitive fields that were leaked (denominator = sensitive fields only)."""
    n_sensitive = sum(1 for k in record if k in SENSITIVE_FIELDS)
    if n_sensitive == 0:
        return 0.0
    return round(len(findings) / n_sensitive, 4)