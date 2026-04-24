from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.detection.leakage_rules import SENSITIVE_FIELDS
from src.evaluation.matrix import build_comparison_matrix, strategy_significance_test


def save_matrices(matrices: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        matrix.to_csv(output_dir / f"{name}_matrix.csv")


def save_results(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_summary(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    agg_dict = {
        "avg_leakage_score": ("leakage_score", "mean"),
        "std_leakage_score": ("leakage_score", "std"),
        "avg_normalized_leakage": ("normalized_leakage_score", "mean"),
        "avg_exposure_rate": ("exposure_rate", "mean"),
        "avg_utility_score": ("utility_score", "mean"),
        "std_utility_score": ("utility_score", "std"),
        "leakage_rate": ("has_leakage", "mean"),
        "avg_put_index": ("put_index", "mean"),
    }

    # Only include columns that exist (guards against partial runs)
    valid_agg = {k: v for k, v in agg_dict.items() if v[0] in df.columns}

    summary = (
        df.groupby(["model", "strategy"], as_index=False)
        .agg(**valid_agg)
        .sort_values(["model", "avg_leakage_score"])
        .round(4)
    )

    summary.to_csv(path, index=False)


def save_field_leakage_breakdown(df: pd.DataFrame, path: Path) -> None:
    """Per-field leakage rate table: which sensitive field leaks most per strategy × model."""
    path.parent.mkdir(parents=True, exist_ok=True)

    field_cols = [f"leaked_{f}" for f in SENSITIVE_FIELDS if f"leaked_{f}" in df.columns]
    if not field_cols:
        return

    breakdown = (
        df.groupby(["model", "strategy"])[field_cols]
        .mean()
        .round(4)
        .reset_index()
    )
    breakdown.columns = [
        c.replace("leaked_", "") if c.startswith("leaked_") else c
        for c in breakdown.columns
    ]
    breakdown.to_csv(path, index=False)


def save_put_scatter(df: pd.DataFrame, path: Path) -> None:
    """
    Privacy-Utility Trade-off scatter plot with standard error bars.
    One point per (model, strategy); error bars show ±1 SE across repetitions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if "utility_score" not in df.columns or "normalized_leakage_score" not in df.columns:
        return

    n_col = "repetition" if "repetition" in df.columns else None
    summary = df.groupby(["model", "strategy"], as_index=False).agg(
        utility=("utility_score", "mean"),
        utility_se=("utility_score", "sem"),
        leakage=("normalized_leakage_score", "mean"),
        leakage_se=("normalized_leakage_score", "sem"),
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^", "D", "v"]
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    strategy_order = sorted(summary["strategy"].unique())

    for i, (model, grp) in enumerate(summary.groupby("model")):
        for j, (_, row) in enumerate(grp.iterrows()):
            strat_idx = strategy_order.index(row["strategy"])
            ax.errorbar(
                row["leakage"], row["utility"],
                xerr=row["leakage_se"],
                yerr=row["utility_se"],
                fmt=markers[strat_idx % len(markers)],
                color=colors[i % len(colors)],
                markersize=9,
                capsize=4,
                label=f"{model} / {row['strategy']}",
            )
            ax.annotate(
                row["strategy"].replace("_", "\n"),
                (row["leakage"], row["utility"]),
                textcoords="offset points",
                xytext=(7, 5),
                fontsize=7,
            )

    ax.set_xlabel("Normalized Leakage Score (lower = better privacy)")
    ax.set_ylabel("Utility Score (higher = better utility)")
    ax.set_title("Privacy–Utility Trade-off by Strategy and Model\n(error bars = ±1 SE across repetitions)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_significance_tables(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["leakage_score", "utility_score", "exposure_rate"]:
        if metric not in df.columns:
            continue
        sig = strategy_significance_test(df, metric)
        sig.to_csv(output_dir / f"kruskal_{metric}.csv", index=False)