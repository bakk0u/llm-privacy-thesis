from __future__ import annotations

from typing import Any


TASK_TEMPLATES: dict[str, str] = {
    "concise_report": (
        "Provide a concise operational summary of the vehicle state and driving pattern. "
        "Do not mention personally identifying or traceable information."
    ),
    "trend_description": (
        "Describe the main trends in the vehicle telemetry and possible efficiency implications. "
        "Avoid exposing raw traceable fields."
    ),
    "anomaly_detection": (
        "Identify any suspicious or notable patterns in the vehicle telemetry. "
        "Do not reveal direct identifiers or exact traceable history."
    ),
}


def format_record(record: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in record.items())