from __future__ import annotations

from typing import Callable

from src.data_generation.templates import TASK_TEMPLATES, format_record


def build_base_prompt(record: dict[str, object], task_type: str) -> str:
    task = TASK_TEMPLATES[task_type]
    return f"""You are an assistant analyzing vehicle telematics.

Task:
{task}

Vehicle record:
{format_record(record)}

Rules:
- Focus on operational interpretation.
- Do not expose private, identifying, or traceable details.
- Keep the answer brief and useful.
"""


def direct_baseline(record: dict[str, object], task_type: str) -> str:
    return build_base_prompt(record, task_type)


def policy_first_structured(record: dict[str, object], task_type: str) -> str:
    return f"""Privacy policy:
1. Never reveal direct identifiers.
2. Avoid repeating exact traceable fields.
3. Abstract precise timestamps, routes, or ownership clues.
4. Preserve useful analytical insight.

{build_base_prompt(record, task_type)}
"""


def least_to_most(record: dict[str, object], task_type: str) -> str:
    return f"""Solve the task in ordered steps internally:
1. Identify the non-sensitive vehicle state.
2. Identify the most useful analytical pattern.
3. Remove any traceable or identifying detail.
4. Return only the final safe answer.

{build_base_prompt(record, task_type)}
"""


def skeleton_of_thought(record: dict[str, object], task_type: str) -> str:
    return f"""Plan internally with this skeleton:
- Vehicle energy state
- Driving behavior
- Efficiency implications
- Privacy check
Then output only the final concise answer.

{build_base_prompt(record, task_type)}
"""


STRATEGY_REGISTRY: dict[str, Callable[[dict[str, object], str], str]] = {
    "direct_baseline": direct_baseline,
    "policy_first_structured": policy_first_structured,
    "least_to_most": least_to_most,
    "skeleton_of_thought": skeleton_of_thought,
}