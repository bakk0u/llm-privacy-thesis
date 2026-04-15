TELEMATICS_COLUMNS = [
    "timestamp",
    "batteryvoltage",
    "sochighlevel",
    "energyconsumptionaverage",
    "electricremaining",
    "mileage",
    "speed_kmh",
    "acceleration_kmh2",
]

SENSITIVE_COLUMNS = {
    "timestamp",
    "mileage",
    "vin",
    "device_id",
    "latitude",
    "longitude",
    "email",
    "phone",
    "full_name",
}