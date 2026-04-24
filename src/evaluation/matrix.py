from __future__ import annotations

import pandas as pd
from scipy.stats import kruskal


def build_comparison_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Strategy × Model matrix with mean ± std per cell.

    Returns a MultiIndex column DataFrame:
        level 0 = model name
        level 1 = "mean" | "std"
    A 'rank' column (by mean across models) is appended per model.
    """
    mean_mat = df.pivot_table(
        index="strategy",
        columns="model",
        values=metric,
        aggfunc="mean",
    ).round(4)

    std_mat = df.pivot_table(
        index="strategy",
        columns="model",
        values=metric,
        aggfunc="std",
    ).round(4)

    mean_mat.columns = pd.MultiIndex.from_product([mean_mat.columns, ["mean"]])
    std_mat.columns = pd.MultiIndex.from_product([std_mat.columns, ["std"]])

    combined = pd.concat([mean_mat, std_mat], axis=1).sort_index(axis=1)
    return combined


def strategy_significance_test(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Kruskal-Wallis H-test across strategies, run separately per model.

    Returns a DataFrame with columns: model, kruskal_H, p_value, significant_0.05
    A significant result means at least one strategy differs from the others.
    """
    records = []
    for model, model_df in df.groupby("model"):
        groups = [
            grp[metric].dropna().values
            for _, grp in model_df.groupby("strategy")
            if len(grp[metric].dropna()) > 0
        ]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        records.append({
            "model": model,
            "kruskal_H": round(stat, 4),
            "p_value": round(p, 4),
            "significant_0.05": p < 0.05,
        })
    return pd.DataFrame(records)