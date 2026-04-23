from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT, load_experiment_config
from src.data_generation.schema import TELEMATICS_COLUMNS


def load_dataset():
    df = pd.read_csv("data/tesla.csv")

    # --- Add synthetic sensitive fields ---
    df["driver_id"] = [f"DRV{i:05d}" for i in range(len(df))]
    df["email"] = [f"user{i}@example.com" for i in range(len(df))]
    df["location"] = "Bremen"

    # --- Generate ground truth ---
    def make_ground_truth(row):
        speed = row["speed_kmh"]
        accel = row["acceleration_kmh2"]
        energy = row["electricenergyremaining"]

        if speed > 100:
            speed_desc = "high-speed driving"
        elif speed > 50:
            speed_desc = "moderate-speed driving"
        else:
            speed_desc = "low-speed driving"

        if accel > 20:
            accel_desc = "with strong acceleration"
        elif accel > 5:
            accel_desc = "with moderate acceleration"
        else:
            accel_desc = "with stable acceleration"

        return f"The vehicle shows {speed_desc} {accel_desc}, with remaining electric energy at {energy}."

    df["ground_truth"] = df.apply(make_ground_truth, axis=1)

    return df


def sample_dataset(df: pd.DataFrame) -> pd.DataFrame:
    config = load_experiment_config()
    max_records = config.get("max_records")
    sample_size = config.get("sample_size", 25)
    random_seed = config.get("random_seed", 42)

    if max_records:
        return df.head(int(max_records)).copy()

    if len(df) <= sample_size:
        return df.copy()

    return df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)