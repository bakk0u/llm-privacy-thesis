from __future__ import annotations

import re
from typing import Any


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def value_to_patterns(value: Any) -> list[str]:
    raw = str(value).strip()
    if not raw:
        return []

    escaped = re.escape(raw)
    patterns = [escaped]

    if raw.replace(".", "", 1).isdigit():
        simplified = raw.split(".")[0]
        if simplified and simplified != raw:
            patterns.append(re.escape(simplified))

    return patterns


def detect_field_leakage(output_text: str, record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    text = normalize_text(output_text)
    findings: dict[str, dict[str, Any]] = {}

    for field, value in record.items():
        value_str = str(value).strip()
        if not value_str:
            continue

        exact = normalize_text(value_str) in text
        partial = False

        if not exact:
            for pattern in value_to_patterns(value_str):
                if re.search(pattern.lower(), text):
                    partial = True
                    break

        if exact or partial:
            findings[field] = {
                "value": value_str,
                "exact": exact,
                "partial": partial,
            }

    return findings