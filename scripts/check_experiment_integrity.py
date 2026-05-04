from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "run_id",
    "record_id",
    "model",
    "strategy",
    "prompt",
    "response",
    "leakage_score",
    "normalized_leakage_score",
    "exposure_rate",
    "utility_score",
    "timestamp",
}


def _load_run(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "raw" / "results.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing raw results: {results_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = pd.read_csv(results_path)
    return manifest, results


def check_run(run_dir: Path) -> int:
    try:
        manifest, results = _load_run(run_dir)
    except FileNotFoundError as exc:
        print(f"Run directory: {run_dir}")
        print(f"ERROR: {exc}")
        print("STATUS: NOT READY")
        return 1

    models = list(manifest.get("models", []))
    experiment = manifest.get("experiment_config", {})
    strategies = list(experiment.get("strategies", []))
    repetitions = int(experiment.get("n_repetitions", 1))
    n_records = int(manifest.get("n_records", results["record_id"].nunique()))
    expected_rows = int(manifest.get("n_rows_total", len(models) * len(strategies) * n_records * repetitions))

    missing_columns = sorted(REQUIRED_COLUMNS - set(results.columns))
    actual_rows = len(results)
    empty_responses = int(results["response"].fillna("").astype(str).str.strip().eq("").sum()) if "response" in results else actual_rows

    key_cols = ["model", "strategy", "record_id", "repetition"]
    duplicate_rows = int(results.duplicated(subset=key_cols).sum()) if set(key_cols).issubset(results.columns) else actual_rows

    missing_combinations: list[tuple[str, str, int, int]] = []
    if set(key_cols).issubset(results.columns):
        observed = set(results[key_cols].itertuples(index=False, name=None))
        record_ids = sorted(results["record_id"].dropna().unique().tolist())
        expected = set(product(models, strategies, record_ids, range(repetitions)))
        missing_combinations = sorted(expected - observed)

    ready = (
        actual_rows == expected_rows
        and not missing_columns
        and empty_responses == 0
        and duplicate_rows == 0
        and len(missing_combinations) == 0
    )

    print(f"Run directory: {run_dir}")
    print(f"Expected row count: {expected_rows}")
    print(f"Actual row count: {actual_rows}")
    print(f"Missing required columns: {len(missing_columns)}")
    if missing_columns:
        print("  " + ", ".join(missing_columns))
    print(f"Missing model/strategy/record combinations: {len(missing_combinations)}")
    for combo in missing_combinations[:20]:
        print(f"  model={combo[0]} strategy={combo[1]} record_id={combo[2]} repetition={combo[3]}")
    if len(missing_combinations) > 20:
        print(f"  ... {len(missing_combinations) - 20} more")
    print(f"Empty responses: {empty_responses}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"STATUS: {'READY' if ready else 'NOT READY'}")
    return 0 if ready else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a completed experiment run.")
    parser.add_argument("run_dir", type=Path, help="Path to results/runs/<run_id>")
    args = parser.parse_args()
    return check_run(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
