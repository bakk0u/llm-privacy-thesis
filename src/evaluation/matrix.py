import pandas as pd

def build_comparison_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Creates a matrix:
        rows = strategies
        cols = models
        values = average metric
    """

    matrix = df.pivot_table(
        index="strategy",
        columns="model",
        values=metric,
        aggfunc="mean"
    )

    return matrix