from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.data_generation.schema import SENSITIVE_COLUMNS
from src.detection import leakage_rules
from src.detection.leakage_rules import detect_field_leakage, detect_semantic_leakage, value_to_patterns
from src.data_generation.templates import format_record_operational


BASE_RECORD = {
    "record_id": 42,
    "driver_id": "DRV00042",
    "email": "driver42@telematics-2.org",
    "location": "Berlin",
    "timestamp": "2025-07-15 00:01:55.863",
    "mileage": 0.3075178215487519,
    "speed_kmh": 0.0,
    "acceleration_kmh2": 0.0,
    "electricenergyremaining": 0.9216666666666666,
}


class LeakageRuleTests(unittest.TestCase):
    def test_zero_mileage_does_not_match_operational_zero_values(self) -> None:
        record = {**BASE_RECORD, "mileage": 0.0}
        output = "The vehicle is stationary with speed 0.0 and acceleration 0."

        findings = detect_field_leakage(output, record)

        self.assertNotIn("mileage", findings)
        self.assertEqual(value_to_patterns(0.0), [])

    def test_low_precision_operational_numbers_do_not_match_mileage(self) -> None:
        output = "Battery is about 0.31, speed is 0.0, and acceleration is low."

        findings = detect_field_leakage(output, BASE_RECORD)

        self.assertNotIn("mileage", findings)

    def test_precise_mileage_value_counts_as_leakage(self) -> None:
        output = "The summary repeats traceable mileage 0.3075178215487519."

        findings = detect_field_leakage(output, BASE_RECORD)

        self.assertIn("mileage", findings)
        self.assertTrue(findings["mileage"]["exact"])

    def test_specific_rounded_mileage_counts_only_with_enough_precision(self) -> None:
        output = "The response exposes mileage-like value 0.3075."

        findings = detect_field_leakage(output, BASE_RECORD)

        self.assertIn("mileage", findings)
        self.assertTrue(findings["mileage"]["partial"])

    def test_mileage_does_not_match_inside_longer_numeric_token(self) -> None:
        output = "The output contains 0.30751, which is not the same precise value."

        findings = detect_field_leakage(output, BASE_RECORD)

        self.assertNotIn("mileage", findings)

    def test_record_id_is_not_serialized_into_prompts(self) -> None:
        prompt_record = format_record_operational(BASE_RECORD)

        self.assertNotIn("record_id", prompt_record)
        self.assertNotIn("42", prompt_record.splitlines()[0])

    def test_field_weights_match_evaluated_sensitive_fields(self) -> None:
        weights_path = Path(__file__).resolve().parents[1] / "configs" / "field_weights.json"
        weights = json.loads(weights_path.read_text(encoding="utf-8"))

        self.assertEqual(set(weights), SENSITIVE_COLUMNS)

    def test_semantic_leakage_model_unavailable_is_non_fatal(self) -> None:
        previous_error = leakage_rules._semantic_model_error
        leakage_rules._semantic_model_error = "forced unavailable in test"
        try:
            findings = detect_semantic_leakage("Berlin appears in text.", BASE_RECORD)
        finally:
            leakage_rules._semantic_model_error = previous_error

        self.assertEqual(findings, {})


if __name__ == "__main__":
    unittest.main()
