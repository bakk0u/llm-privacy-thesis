from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "run_id",
    "record_id",
    "model",
    "strategy",
    "repetition",
    "prompt",
    "raw_output",
    "ground_truth",
    "leakage_score",
    "normalized_leakage_score",
    "exposure_rate",
    "utility_score",
    "put_index",
    "exact_leaks",
    "partial_leaks",
    "semantic_leaks",
    "leaked_fields",
}

KEY_COLUMNS = ["model", "strategy", "record_id", "repetition"]
SUMMARY_TOLERANCE = 1e-4


def _load_run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "raw" / "results.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing raw results: {results_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = pd.read_csv(results_path)
    return manifest, results


def _blank_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return len(df)
    return int(df[column].fillna("").astype(str).str.strip().eq("").sum())


def _expected_values(manifest: dict[str, Any], results: pd.DataFrame) -> tuple[list[str], list[str], list[int], int, int]:
    models = list(manifest.get("models", []))
    experiment = manifest.get("experiment_config", {})
    strategies = list(experiment.get("strategies", []))
    repetitions = int(experiment.get("n_repetitions", 1))

    record_ids = manifest.get("record_ids")
    if record_ids:
        record_ids = [int(record_id) for record_id in record_ids]
    elif "record_id" in results.columns:
        record_ids = sorted(int(record_id) for record_id in results["record_id"].dropna().unique())
    else:
        record_ids = []

    expected_rows = manifest.get("expected_rows", manifest.get("n_rows_total"))
    if expected_rows is None:
        expected_rows = len(models) * len(strategies) * len(record_ids) * repetitions
    expected_rows = int(expected_rows)

    return models, strategies, record_ids, repetitions, expected_rows


def _missing_combinations(
    results: pd.DataFrame,
    models: list[str],
    strategies: list[str],
    record_ids: list[int],
    repetitions: int,
) -> list[tuple[str, str, int, int]]:
    if not set(KEY_COLUMNS).issubset(results.columns):
        return []

    observed = set(results[KEY_COLUMNS].itertuples(index=False, name=None))
    expected = set(product(models, strategies, record_ids, range(repetitions)))
    return sorted(expected - observed)


def _summary_expected(results: pd.DataFrame) -> pd.DataFrame:
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
    valid_agg = {name: spec for name, spec in agg_dict.items() if spec[0] in results.columns}
    if not valid_agg:
        return pd.DataFrame()
    return (
        results.groupby(["model", "strategy"], as_index=False)
        .agg(**valid_agg)
        .round(4)
        .sort_values(["model", "strategy"])
        .reset_index(drop=True)
    )


def _compare_group_table(expected: pd.DataFrame, actual: pd.DataFrame) -> list[str]:
    if expected.empty:
        return []
    required_keys = {"model", "strategy"}
    if not required_keys.issubset(actual.columns):
        return ["missing model/strategy columns"]

    compare_cols = [
        col for col in expected.columns
        if col not in required_keys and col in actual.columns
    ]
    missing_metric_cols = [
        col for col in expected.columns
        if col not in required_keys and col not in actual.columns
    ]
    merged = expected.merge(actual, on=["model", "strategy"], suffixes=("_expected", "_actual"), how="outer", indicator=True)
    problems: list[str] = []

    for col in missing_metric_cols:
        problems.append(f"missing aggregate column: {col}")

    missing_rows = merged[merged["_merge"] != "both"]
    for _, row in missing_rows.iterrows():
        problems.append(f"row mismatch model={row.get('model')} strategy={row.get('strategy')}")

    both = merged[merged["_merge"] == "both"]
    for _, row in both.iterrows():
        for col in compare_cols:
            expected_value = row[f"{col}_expected"]
            actual_value = row[f"{col}_actual"]
            if pd.isna(expected_value) and pd.isna(actual_value):
                continue
            if pd.isna(expected_value) != pd.isna(actual_value):
                problems.append(f"{row['model']} / {row['strategy']} / {col}: NaN mismatch")
                continue
            if abs(float(expected_value) - float(actual_value)) > SUMMARY_TOLERANCE:
                problems.append(
                    f"{row['model']} / {row['strategy']} / {col}: "
                    f"expected {expected_value}, found {actual_value}"
                )
    return problems


def _check_summary_table(results: pd.DataFrame, run_dir: Path, complete_run: bool) -> tuple[str, list[str]]:
    path = run_dir / "tables" / "summary_metrics.csv"
    if not path.exists():
        if complete_run:
            return "missing", ["summary_metrics.csv is missing"]
        return "skipped", []

    actual = pd.read_csv(path)
    expected = _summary_expected(results)
    problems = _compare_group_table(expected, actual)
    return ("ok" if not problems else "mismatch"), problems


def _field_breakdown_expected(results: pd.DataFrame) -> pd.DataFrame:
    field_cols = sorted(col for col in results.columns if col.startswith("leaked_"))
    if not field_cols:
        return pd.DataFrame()
    breakdown = (
        results.groupby(["model", "strategy"], as_index=False)[field_cols]
        .mean()
        .round(4)
        .sort_values(["model", "strategy"])
        .reset_index(drop=True)
    )
    return breakdown.rename(columns={col: col.replace("leaked_", "") for col in field_cols})


def _check_field_breakdown(results: pd.DataFrame, run_dir: Path, complete_run: bool) -> tuple[str, list[str]]:
    path = run_dir / "tables" / "field_leakage_breakdown.csv"
    if not path.exists():
        if complete_run:
            return "missing", ["field_leakage_breakdown.csv is missing"]
        return "skipped", []

    actual = pd.read_csv(path)
    expected = _field_breakdown_expected(results)
    problems = _compare_group_table(expected, actual)
    return ("ok" if not problems else "mismatch"), problems


def _check_matrices(results: pd.DataFrame, run_dir: Path, complete_run: bool) -> tuple[str, list[str]]:
    metrics = [
        "leakage_score",
        "normalized_leakage_score",
        "exposure_rate",
        "utility_score",
        "put_index",
    ]
    problems: list[str] = []
    checked_any = False

    for metric in metrics:
        if metric not in results.columns:
            continue

        path = run_dir / "tables" / "matrices" / f"{metric}_matrix.csv"
        if not path.exists():
            if complete_run:
                problems.append(f"{metric}_matrix.csv is missing")
            continue

        checked_any = True
        actual = pd.read_csv(path, header=[0, 1], index_col=0)
        expected_mean = results.pivot_table(
            index="strategy",
            columns="model",
            values=metric,
            aggfunc="mean",
        ).round(4)
        expected_std = results.pivot_table(
            index="strategy",
            columns="model",
            values=metric,
            aggfunc="std",
        ).round(4)

        for strategy in expected_mean.index:
            if strategy not in actual.index:
                problems.append(f"{metric}: missing strategy row {strategy}")
                continue
            for model in expected_mean.columns:
                for kind, expected_table in (("mean", expected_mean), ("std", expected_std)):
                    column = (model, kind)
                    if column not in actual.columns:
                        problems.append(f"{metric}: missing matrix column {model}/{kind}")
                        continue
                    expected_value = expected_table.loc[strategy, model]
                    actual_value = actual.loc[strategy, column]
                    if pd.isna(expected_value) and pd.isna(actual_value):
                        continue
                    if pd.isna(expected_value) != pd.isna(actual_value):
                        problems.append(f"{metric}: {strategy}/{model}/{kind} NaN mismatch")
                        continue
                    if abs(float(expected_value) - float(actual_value)) > SUMMARY_TOLERANCE:
                        problems.append(
                            f"{metric}: {strategy}/{model}/{kind}: "
                            f"expected {expected_value}, found {actual_value}"
                        )

    if not problems:
        return ("ok" if checked_any or complete_run else "skipped"), []
    return "mismatch", problems


def check_run(run_dir: Path) -> int:
    try:
        manifest, results = _load_run(run_dir)
    except FileNotFoundError as exc:
        print(f"Run directory: {run_dir}")
        print(f"ERROR: {exc}")
        print("STATUS: NOT READY")
        return 1

    models, strategies, record_ids, repetitions, expected_rows = _expected_values(manifest, results)
    actual_rows = len(results)
    missing_columns = sorted(REQUIRED_COLUMNS - set(results.columns))
    missing_prompts = _blank_count(results, "prompt")
    missing_raw_outputs = _blank_count(results, "raw_output")
    duplicate_rows = (
        int(results.duplicated(subset=KEY_COLUMNS).sum())
        if set(KEY_COLUMNS).issubset(results.columns)
        else actual_rows
    )
    missing_combinations = _missing_combinations(results, models, strategies, record_ids, repetitions)
    complete_run = actual_rows == expected_rows and len(missing_combinations) == 0

    summary_status, summary_problems = _check_summary_table(results, run_dir, complete_run)
    field_status, field_problems = _check_field_breakdown(results, run_dir, complete_run)
    matrix_status, matrix_problems = _check_matrices(results, run_dir, complete_run)

    ready = (
        complete_run
        and not missing_columns
        and missing_prompts == 0
        and missing_raw_outputs == 0
        and duplicate_rows == 0
        and not summary_problems
        and not field_problems
        and not matrix_problems
    )

    partial = actual_rows < expected_rows or manifest.get("status") in {"running", "interrupted"}
    status = "READY" if ready else ("PARTIAL" if partial and actual_rows > 0 else "NOT READY")

    print(f"Run directory: {run_dir}")
    print(f"Manifest status: {manifest.get('status', 'unknown')}")
    print(f"Expected row count: {expected_rows}")
    print(f"Actual row count: {actual_rows}")
    print(f"Progress: {actual_rows}/{expected_rows}")
    print(f"Models: {len(models)}")
    print(f"Strategies: {len(strategies)}")
    print(f"Records: {len(record_ids)}")
    print(f"Repetitions: {repetitions}")
    print(f"Missing required columns: {len(missing_columns)}")
    if missing_columns:
        print("  " + ", ".join(missing_columns))
    print(f"Missing model/strategy/record/repetition combinations: {len(missing_combinations)}")
    for combo in missing_combinations[:20]:
        print(f"  model={combo[0]} strategy={combo[1]} record_id={combo[2]} repetition={combo[3]}")
    if len(missing_combinations) > 20:
        print(f"  ... {len(missing_combinations) - 20} more")
    print(f"Duplicate combinations: {duplicate_rows}")
    print(f"Missing prompts: {missing_prompts}")
    print(f"Missing raw outputs: {missing_raw_outputs}")
    print(f"Summary table consistency: {summary_status}")
    for problem in summary_problems[:20]:
        print(f"  {problem}")
    if len(summary_problems) > 20:
        print(f"  ... {len(summary_problems) - 20} more")
    print(f"Field leakage breakdown consistency: {field_status}")
    for problem in field_problems[:20]:
        print(f"  {problem}")
    if len(field_problems) > 20:
        print(f"  ... {len(field_problems) - 20} more")
    print(f"Matrix table consistency: {matrix_status}")
    for problem in matrix_problems[:20]:
        print(f"  {problem}")
    if len(matrix_problems) > 20:
        print(f"  ... {len(matrix_problems) - 20} more")
    print(f"STATUS: {status}")
    return 0 if ready else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a completed or partial experiment run.")
    parser.add_argument("run_dir", type=Path, help="Path to results/runs/<run_id>")
    args = parser.parse_args()
    return check_run(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
