from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT, load_experiment_config, load_field_weights, load_model_configs
from src.data_generation.generator import load_dataset, sample_dataset
from src.models.ollama_client import OllamaClient
from src.prompting.base import STRATEGY_REGISTRY
from src.detection.leakage_rules import detect_field_leakage
from src.detection.scoring import compute_exposure_rate, compute_leakage_score
from src.detection.utility import utility_score
from src.evaluation.export import save_results, save_summary


def run_full_experiment() -> None:
    experiment = load_experiment_config()
    field_weights = load_field_weights()
    models = load_model_configs()

    df = sample_dataset(load_dataset())
    client = OllamaClient()

    rows: list[dict[str, object]] = []

    for model in models:
        for strategy_name in experiment["strategies"]:
            if strategy_name not in STRATEGY_REGISTRY:
                raise KeyError(f"Unknown strategy: {strategy_name}")

            prompt_builder = STRATEGY_REGISTRY[strategy_name]

            for row_id, (_, series) in enumerate(df.iterrows()):
                record = series.to_dict()
                prompt = prompt_builder(record, experiment["task_type"])
                output = client.generate(prompt, model)

                findings = detect_field_leakage(output, record)
                leakage = compute_leakage_score(findings, field_weights)
                exposure = compute_exposure_rate(findings, record)
                util = utility_score(output)

                rows.append({
                    "row_id": row_id,
                    "model": model.name,
                    "strategy": strategy_name,
                    "prompt": prompt,
                    "output": output,
                    "leakage_score": leakage,
                    "exposure_rate": exposure,
                    "utility_score": util,
                    "has_leakage": int(bool(findings)),
                    "leaked_fields": "|".join(findings.keys()),
                })

    results_df = pd.DataFrame(rows)

    raw_path = PROJECT_ROOT / experiment["output_path"]
    summary_path = PROJECT_ROOT / "results" / "tables" / "summary_metrics.csv"

    save_results(results_df, raw_path)
    save_summary(results_df, summary_path)

    print(f"Saved raw results to {raw_path}")
    print(f"Saved summary results to {summary_path}")