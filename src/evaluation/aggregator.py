from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import KEY_TERMS, PROJECT_ROOT, load_experiment_config, load_field_weights, load_model_configs
from src.data_generation.generator import load_dataset, sample_dataset
from src.detection.leakage_rules import (
    SENSITIVE_FIELDS,
    detect_field_leakage,
    detect_semantic_leakage,
    semantic_leakage_enabled,
    semantic_leakage_error,
)
from src.detection.scoring import compute_exposure_rate, compute_leakage_score, compute_normalized_leakage_score
from src.detection.utility import utility_score
from src.evaluation.export import (
    save_field_leakage_breakdown,
    save_matrices,
    save_put_scatter,
    save_results,
    save_significance_tables,
    save_summary,
)
from src.evaluation.matrix import build_comparison_matrix
from src.models.ollama_client import OllamaClient
from src.prompting.base import STRATEGY_REGISTRY

# Privacy-Utility Trade-off weight: how much to penalise leakage vs. reward utility.
_PUT_ALPHA = 1.0
_LARGE_GENERATION_WARNING_THRESHOLD = 10_000

_FIELD_LEAK_COLUMNS = [f"leaked_{field}" for field in sorted(SENSITIVE_FIELDS)]
_RAW_RESULT_COLUMNS = [
    "run_id",
    "record_id",
    "row_id",
    "repetition",
    "driving_state",
    "model",
    "strategy",
    "prompt",
    "raw_output",
    "response",
    "output",
    "ground_truth",
    "leakage_score",
    "normalized_leakage_score",
    "exposure_rate",
    "utility_score",
    "put_index",
    "timestamp",
    "generated_at",
    "has_leakage",
    "has_lexical_leakage",
    "has_semantic_leakage",
    "exact_leaks",
    "partial_leaks",
    "semantic_leaks",
    "leaked_fields",
    "semantic_leaked_fields",
    *_FIELD_LEAK_COLUMNS,
]


def _append_raw_result(row: dict[str, object], path: Path) -> None:
    """Persist one generation immediately so interrupted runs keep progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_RAW_RESULT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_manifest(manifest: dict[str, object], run_dir: Path) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_all_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "leakage_score": build_comparison_matrix(df, "leakage_score"),
        "normalized_leakage_score": build_comparison_matrix(df, "normalized_leakage_score"),
        "exposure_rate": build_comparison_matrix(df, "exposure_rate"),
        "utility_score": build_comparison_matrix(df, "utility_score"),
        "put_index": build_comparison_matrix(df, "put_index"),
    }


def run_full_experiment() -> None:
    experiment = load_experiment_config()
    field_weights = load_field_weights()
    models = load_model_configs()
    n_repetitions: int = int(experiment.get("n_repetitions", 1))

    full_dataset = load_dataset()
    dataset = sample_dataset(full_dataset)
    client = OllamaClient()

    rows: list[dict[str, object]] = []
    total = len(models) * len(experiment["strategies"]) * len(dataset) * n_repetitions
    done = 0
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = PROJECT_ROOT / "results" / "runs" / run_id
    raw_dir = run_dir / "raw"
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    raw_results_path = raw_dir / "results.csv"

    manifest: dict[str, object] = {
        "run_id": run_id,
        "status": "running",
        "experiment_config": experiment,
        "models": [m.name for m in models],
        "semantic_leakage_detection_enabled": semantic_leakage_enabled(),
        "semantic_leakage_detection_error": semantic_leakage_error(),
        "n_records": len(dataset),
        "record_ids": [int(record_id) for record_id in dataset["record_id"].tolist()]
            if "record_id" in dataset.columns else [],
        "expected_rows": total,
        "n_rows_total": total,
        "completed_rows": 0,
        "driving_state_counts": dataset["driving_state"].value_counts().to_dict()
            if "driving_state" in dataset.columns else {},
    }
    _write_manifest(manifest, run_dir)

    print("\nExperiment pre-run summary")
    print(f"Run ID                 : {run_id}")
    print(f"Dataset size           : {len(full_dataset)}")
    print(f"Selected sample size   : {len(dataset)}")
    print(f"Models                 : {len(models)}")
    print(f"Strategies             : {len(experiment['strategies'])}")
    print(f"Repetitions            : {n_repetitions}")
    print(f"Expected generations   : {total}")
    print(f"Output directory       : {run_dir}")
    if total > _LARGE_GENERATION_WARNING_THRESHOLD:
        print(
            "WARNING: expected generations exceed "
            f"{_LARGE_GENERATION_WARNING_THRESHOLD}; this run may take a long time."
        )

    try:
        for model in models:
            for strategy_name in experiment["strategies"]:
                if strategy_name not in STRATEGY_REGISTRY:
                    raise KeyError(f"Unknown strategy: {strategy_name}")

                prompt_builder = STRATEGY_REGISTRY[strategy_name]

                for row_id, (_, series) in enumerate(dataset.iterrows()):
                    record = series.to_dict()
                    prompt = prompt_builder(record, experiment["task_type"])

                    for rep in range(n_repetitions):
                        current = done + 1
                        print(
                            f"[{current}/{total}] model={model.name} | strategy={strategy_name} "
                            f"| row={row_id + 1} | rep={rep + 1}"
                        )

                        output = client.generate(prompt, model)
                        generated_at = datetime.now().isoformat(timespec="seconds")

                        findings = detect_field_leakage(output, record, sensitive_only=True)
                        leakage = compute_leakage_score(findings, field_weights)
                        normalized_leakage = compute_normalized_leakage_score(findings, field_weights)
                        exposure = compute_exposure_rate(findings, record)
                        ground_truth = str(series["ground_truth"])
                        util = utility_score(output, ground_truth, KEY_TERMS)

                        # Privacy-Utility Trade-off index (higher = better combined performance)
                        put_index = util - _PUT_ALPHA * normalized_leakage

                        # Semantic leakage (graceful no-op if sentence-transformers not installed)
                        semantic_findings = detect_semantic_leakage(output, record)
                        semantic_fields = "|".join(semantic_findings.keys())

                        exact_leaks = "|".join(
                            field for field, details in findings.items()
                            if details.get("exact")
                        )
                        partial_leaks = "|".join(
                            field for field, details in findings.items()
                            if details.get("partial")
                        )

                        # Per-field binary leakage flags for breakdown analysis
                        field_flags = {
                            f"leaked_{f}": int(f in findings or f in semantic_findings)
                            for f in SENSITIVE_FIELDS
                        }

                        row = {
                            "run_id": run_id,
                            "record_id": int(record.get("record_id", row_id)),
                            "row_id": row_id,
                            "repetition": rep,
                            "driving_state": record.get("driving_state", "unknown"),
                            "model": model.name,
                            "strategy": strategy_name,
                            "prompt": prompt,
                            "raw_output": output,
                            "response": output,
                            "output": output,
                            "ground_truth": ground_truth,
                            "leakage_score": leakage,
                            "normalized_leakage_score": normalized_leakage,
                            "exposure_rate": exposure,
                            "utility_score": util,
                            "put_index": put_index,
                            "timestamp": generated_at,
                            "generated_at": generated_at,
                            "has_leakage": int(bool(findings) or bool(semantic_findings)),
                            "has_lexical_leakage": int(bool(findings)),
                            "has_semantic_leakage": int(bool(semantic_findings)),
                            "exact_leaks": exact_leaks,
                            "partial_leaks": partial_leaks,
                            "semantic_leaks": json.dumps(semantic_findings, sort_keys=True),
                            "leaked_fields": "|".join(findings.keys()),
                            "semantic_leaked_fields": semantic_fields,
                            **field_flags,
                        }
                        rows.append(row)
                        _append_raw_result(row, raw_results_path)
                        done = current
                        manifest["completed_rows"] = done
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["completed_rows"] = done
        _write_manifest(manifest, run_dir)
        raise

    results_df = pd.DataFrame(rows)

    # Save all outputs under the timestamped run directory
    save_results(results_df, raw_dir / "results.csv")
    save_summary(results_df, tables_dir / "summary_metrics.csv")
    save_field_leakage_breakdown(results_df, tables_dir / "field_leakage_breakdown.csv")
    save_matrices(generate_all_matrices(results_df), tables_dir / "matrices")
    save_put_scatter(results_df, figures_dir / "put_scatter.png")
    save_significance_tables(results_df, tables_dir / "significance")

    # Snapshot the completed state used for reproducibility and integrity checks.
    manifest["status"] = "completed"
    manifest["completed_rows"] = len(results_df)
    _write_manifest(manifest, run_dir)

    print(f"\n{'='*50}")
    print(f"Run ID : {run_id}")
    print(f"Output : {run_dir}")
    print(f"  raw/results.csv")
    print(f"  tables/summary_metrics.csv")
    print(f"  tables/field_leakage_breakdown.csv")
    print(f"  tables/matrices/")
    print(f"  tables/significance/")
    print(f"  figures/put_scatter.png")
    print(f"  run_manifest.json")
    print(f"{'='*50}")
