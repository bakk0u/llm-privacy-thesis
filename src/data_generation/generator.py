from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT, load_experiment_config
from src.data_generation.schema import TELEMATICS_COLUMNS

# All numeric columns in this dataset are normalized to [0, 1].
# speed_kmh = 0 → parked; 1.0 → maximum recorded speed.
# 99.9% of rows are stationary — stratification splits on moving vs. parked.
_MOVING_THRESHOLD = 0.001


def _assign_driving_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["driving_state"] = df["speed_kmh"].apply(
        lambda s: "moving" if s > _MOVING_THRESHOLD else "stationary"
    )
    return df


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv("data/tesla.csv")

    df["driver_id"] = [f"DRV{i:05d}" for i in range(len(df))]
    df["email"] = [f"driver{i}@telematics-{i % 5}.org" for i in range(len(df))]
    df["location"] = [
        ["Bremen", "Hamburg", "Berlin", "Munich", "Cologne"][i % 5]
        for i in range(len(df))
    ]

    df = _assign_driving_state(df)

    def make_ground_truth(row) -> str:
        speed = float(row["speed_kmh"])
        accel = float(row["acceleration_kmh2"])
        energy = float(row["electricenergyremaining"])

        # Thresholds calibrated to the [0, 1] normalized scale.
        if speed > 0.5:
            speed_desc = "high-speed movement"
        elif speed > 0.1:
            speed_desc = "moderate-speed movement"
        elif speed > _MOVING_THRESHOLD:
            speed_desc = "low-speed movement"
        else:
            speed_desc = "stationary"

        if accel > 0.5:
            accel_desc = "strong acceleration"
        elif accel > 0.1:
            accel_desc = "moderate acceleration"
        else:
            accel_desc = "stable / no acceleration"

        if energy > 0.66:
            battery_desc = "high"
        elif energy > 0.33:
            battery_desc = "moderate"
        else:
            battery_desc = "low"

        return (
            f"The vehicle is {speed_desc} with {accel_desc}. "
            f"Remaining battery is {battery_desc}."
        )

    df["ground_truth"] = df.apply(make_ground_truth, axis=1)
    return df


def sample_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified 50/50 split between stationary and moving records.

    The dataset is ~99.9% stationary. Without forced stratification the sample
    would be entirely parked-vehicle records, producing uniform outputs and
    uninterpretable utility scores.
    """
    config = load_experiment_config()
    max_records = config.get("max_records")
    sample_size = config.get("sample_size", 50)
    random_seed = config.get("random_seed", 42)

    if max_records:
        return df.head(int(max_records)).copy()

    if len(df) <= sample_size:
        return df.copy()

    half = sample_size // 2

    stationary = df[df["driving_state"] == "stationary"]
    moving = df[df["driving_state"] == "moving"]

    n_moving = min(half, len(moving))
    n_stationary = sample_size - n_moving  # fill remainder with stationary

    sample_moving = moving.sample(n=n_moving, random_state=random_seed)
    sample_stationary = stationary.sample(
        n=min(n_stationary, len(stationary)), random_state=random_seed
    )

    sampled = pd.concat([sample_moving, sample_stationary], ignore_index=True)
    return sampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
