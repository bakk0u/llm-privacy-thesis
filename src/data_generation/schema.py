"""Shared field definitions for the telematics privacy experiment.

The sensitive-field set below is intentionally limited to fields that are
actually present after data loading and evaluated by the leakage detector.
Operational telemetry such as speed, acceleration, battery, and distance is
not scored as privacy leakage because the task explicitly asks the model to
summarize those values abstractly.
"""

TELEMATICS_COLUMNS = [
    "timestamp",
    "electricenergyremaining",
    "electricremaining",
    "speed_kmh",
    "acceleration_kmh2",
    "mileage",
    "vehiclespeed",
    "distance_m",
]

# Evaluated sensitive/traceable fields:
# - driver_id and email are synthetic direct identifiers.
# - location is a synthetic quasi-identifier.
# - timestamp and mileage are traceable telemetry fields.
# record_id is deliberately excluded from prompts, so it is not evaluated.
SENSITIVE_COLUMNS = {
    "driver_id",
    "email",
    "location",
    "timestamp",
    "mileage",
}
