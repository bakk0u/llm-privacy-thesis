# Mitigating Privacy Leakage in LLMs for Vehicular Telematics

Empirical master's thesis repository for evaluating whether prompt structure can reduce privacy leakage in LLM-generated summaries of vehicular telematics while preserving operational utility.

## Thesis Topic

**Mitigating Privacy Leakage in Large Language Models through Structured Prompting: An Empirical Study on Vehicular Telematics**

This project implements a controlled experimental pipeline for comparing prompt strategies on the same vehicle-record summarisation task. Each generation is evaluated against programmatically generated ground truth and sensitive-field leakage rules.

All thesis claims in this repository are based on the completed final run only:

```text
results/runs/2026-05-07_19-36-26
```

## Final Run

The completed final run used:

- Run ID: `2026-05-07_19-36-26`
- Records: 700 telematics records
- Models: 2 local Ollama models
- Prompting strategies: 5
- Repetitions: 1
- Total generations: 7,000
- Sampling: 350 moving records and 350 stationary records
- Dataset path expected locally: `data/tesla.csv`

The final run artifacts are committed under `results/runs/2026-05-07_19-36-26/`, including the manifest, raw generations, summary tables, field-leakage breakdowns, matrix tables, significance tables, and the privacy-utility scatter plot.

## Models

The final run evaluates:

- `llama3.1:8b`
- `deepseek-r1:8b`

Both are configured through `configs/model_config.json` and are expected to run locally through Ollama.

## Prompting Strategies

Five strategies are evaluated:

- `no_instruction_control`: task-only control condition.
- `direct_baseline`: direct task prompt with brief privacy instruction.
- `policy_first_structured`: explicit privacy policy before the task.
- `least_to_most`: staged reasoning from observations to final privacy-aware answer.
- `skeleton_of_thought`: structured response skeleton with a final privacy check.

## Metrics

The evaluation reports:

- `leakage_score`: weighted sensitive-field leakage score.
- `normalized_leakage_score`: leakage score normalized by the maximum possible score.
- `exposure_rate`: fraction of sensitive fields exposed.
- `utility_score`: rule-based utility score against generated ground truth.
- `put_index`: privacy-utility trade-off index combining utility and normalized leakage.

Sensitive fields evaluated by the leakage detector are `driver_id`, `email`, `location`, `timestamp`, and `mileage`. Operational fields such as speed, acceleration, and battery level are task-relevant and are not scored as privacy leakage.

## Results Summary

The final run shows very low measured leakage across almost all model-strategy combinations. In the summary table, nine of ten model-strategy combinations have zero average measured leakage. The only non-zero aggregate leakage appears for `deepseek-r1:8b` with `policy_first_structured`, where the average leakage score is very small (`0.0007`) and the leakage rate is `0.0014`.

Utility varies more visibly than leakage in the final run. The highest average utility scores are:

- `deepseek-r1:8b` with `direct_baseline`: `0.4627`
- `llama3.1:8b` with `skeleton_of_thought`: `0.4550`

These findings should be read cautiously. The final results support the narrower claim that, under this dataset sample, prompt set, leakage detector, and local Ollama configuration, the evaluated strategies generally avoided explicit sensitive-field disclosure while preserving moderate operational utility. They do not prove that any prompt strategy prevents privacy leakage in general, nor that the same ranking will hold for other datasets, models, telemetry schemas, or adversarial settings.

## Repository Structure

```text
.
|-- configs/
|   |-- experiment_config.json    # Dataset path, sample size, task, strategies
|   |-- field_weights.json        # Leakage scoring weights
|   `-- model_config.json         # Ollama model configuration
|-- data/
|   `-- tesla.csv                 # Local dataset path; not tracked by git
|-- docs/
|   `-- methodology.md            # Methodology notes
|-- notebooks/
|   `-- supervisor_progress_demo.ipynb
|-- results/
|   `-- runs/
|       `-- 2026-05-07_19-36-26/  # Completed final run used for thesis claims
|           |-- figures/
|           |-- raw/
|           |-- tables/
|           `-- run_manifest.json
|-- scripts/
|   `-- check_experiment_integrity.py
|-- src/
|   |-- data_generation/          # Dataset loading, prompt record formatting, ground truth
|   |-- detection/                # Leakage and utility scoring
|   |-- evaluation/               # Experiment loop, aggregation, matrix export
|   |-- models/                   # Ollama client
|   |-- prompting/                # Prompt strategy definitions
|   |-- config.py
|   `-- main.py
|-- tests/
|   `-- test_leakage_rules.py
|-- requirements.txt
`-- README.md
```

## Data

The repository expects the dataset at:

```text
data/tesla.csv
```

The local working copy may contain this file, but it is not tracked by git. To reproduce the experiment from a fresh clone, create the `data/` directory and place the Tesla telematics CSV at `data/tesla.csv`.

Do not rename the file unless you also update the loader in `src/data_generation/generator.py`. At present, the loader reads `data/tesla.csv` directly.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install and start Ollama, then pull the required local models:

```powershell
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
```

Confirm the dataset is available:

```powershell
Test-Path data\tesla.csv
```

## Run

Run the experiment:

```powershell
python -m src.main
```

Outputs are written under:

```text
results/runs/<run_id>/
```

The active files in `configs/` may be changed for new experiments. The completed thesis run parameters are fixed by `results/runs/2026-05-07_19-36-26/run_manifest.json`, not by later configuration edits.

The final thesis run already committed in this repository is:

```text
results/runs/2026-05-07_19-36-26/
```

## Integrity Check

Validate the completed final run with:

```powershell
python scripts/check_experiment_integrity.py results/runs/2026-05-07_19-36-26
```

Expected status:

```text
STATUS: READY
```

## Limitations

- The final run uses one dataset sample, two local 8B models, five prompting strategies, and one repetition per model-strategy-record combination.
- The dataset is treated as a vehicular telematics case study; conclusions should not be generalized to all connected-vehicle datasets.
- Ground truth is generated programmatically from normalized telemetry fields, which improves reproducibility but may not capture every valid human interpretation of utility.
- Sensitive-field leakage detection is based on configured exact, partial, and semantic checks for known fields. It may miss broader inferential or contextual privacy leakage.
- Synthetic identifiers and locations create controlled privacy targets, but they are not equivalent to naturally occurring personal data risk.
- Ollama model behavior can vary with model version, runtime, hardware, and local configuration.
- The prompt templates exclude direct identifiers from the model input; the experiment therefore measures leakage risk within this controlled prompt representation, not unrestricted raw-record disclosure.

## Supervisor Review Pointers

For review, start with:

- `results/runs/2026-05-07_19-36-26/run_manifest.json`
- `results/runs/2026-05-07_19-36-26/tables/summary_metrics.csv`
- `results/runs/2026-05-07_19-36-26/tables/field_leakage_breakdown.csv`
- `results/runs/2026-05-07_19-36-26/figures/put_scatter.png`

These files document the completed final run used for the thesis analysis.
