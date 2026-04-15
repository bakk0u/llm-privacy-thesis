from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT, load_experiment_config
from src.data_generation.schema import TELEMATICS_COLUMNS


def load_dataset() -> pd.DataFrame:
    config = load_experiment_config()
    dataset_path = PROJECT_ROOT / config["dataset_path"]

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Put your telematics CSV there or update configs/experiment_config.json."
        )

    df = pd.read_csv(dataset_path)
    keep = [col for col in TELEMATICS_COLUMNS if col in df.columns]

    if not keep:
        raise ValueError("None of the expected telematics columns were found in the CSV.")

    return df[keep].dropna().reset_index(drop=True)


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