from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_results(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_summary(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby(["model", "strategy"], as_index=False)
        .agg(
            avg_leakage_score=("leakage_score", "mean"),
            avg_exposure_rate=("exposure_rate", "mean"),
            avg_utility_score=("utility_score", "mean"),
            leakage_rate=("has_leakage", "mean"),
        )
        .sort_values(["model", "avg_leakage_score"])
    )

    summary.to_csv(path, index=False)