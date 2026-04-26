# Mitigating Privacy Leakage in LLMs for Vehicular Telematics

Empirical master's thesis project studying whether structured prompting can reduce privacy leakage in LLM-generated summaries of vehicle telemetry while preserving operational utility.

## Thesis Topic

**Mitigating Privacy Leakage in Large Language Models through Structured Prompting: An Empirical Study on Vehicular Telematics**

This repository implements a controlled experimental framework for evaluating privacy-aware prompt engineering. The central research question is whether structured prompting strategies can reduce the disclosure of sensitive or traceable information in LLM-generated summaries of vehicular telematics data.

The project treats prompting as an empirical intervention: each strategy receives the same underlying task and vehicle record representation, then outputs are evaluated against programmatically generated ground truth and sensitive-field leakage rules.

## Experimental Framework

The framework combines:

- Vehicular telemetry records used as the operational input domain.
- Synthetic sensitive-field injection to create controlled privacy risks.
- Programmatic ground-truth generation for repeatable utility evaluation.
- Local LLM inference through Ollama.
- Multi-strategy prompting evaluation across the same task and dataset sample.
- Leakage, utility, and privacy-utility trade-off metrics.

The current experiment uses 50 records, 3 repetitions per model-strategy pair, and 1,500 total generations.

## Prompting Strategies

Five prompting strategies are evaluated:

- `no_instruction_control`: control condition with no privacy instruction.
- `direct_baseline`: direct task prompt with brief privacy rules.
- `policy_first_structured`: explicit privacy policy before the task.
- `least_to_most`: staged decomposition from lower-risk observations to a final privacy review.
- `skeleton_of_thought`: structured answer skeleton with an explicit privacy gate before final output.

## Evaluated Models

The current experiments run locally through Ollama using:

- `llama3.1:8b`
- `deepseek-r1:8b`

## Metrics

The evaluation reports:

- `leakage_score`: weighted score for exact and partial sensitive-field exposure.
- `normalized_leakage`: leakage score normalized by the maximum possible sensitive-field score.
- `exposure_rate`: fraction of sensitive fields exposed in a generation.
- `utility_score`: rule-based score for preserving task-relevant operational insight.
- PUT index: privacy-utility trade-off score combining utility with normalized leakage.

## Current Research Insight

The current results indicate that structured prompting materially affects the privacy-utility trade-off. In the observed runs, `skeleton_of_thought` gives the strongest trade-off for both evaluated models, combining very low leakage with competitive utility.

DeepSeek leaks less than LLaMA in this setup across the reported aggregate leakage metrics. Field-level results also show that explicit PII such as email and driver ID is easier to suppress than implicit telemetry signals. Mileage is the dominant residual leakage channel, with timestamp leakage appearing mainly in weaker or less structured prompting conditions.

These findings should be read as experiment-specific evidence, not a universal claim about all models, datasets, or privacy threats.

## Repository Structure

```text
.
|-- configs/                  # Experiment, model, and field-weight configuration
|-- docs/                     # Methodology notes
|-- notebooks/                # Supervisor-facing progress and analysis notebook
|-- results/runs/             # Generated experiment outputs
|   `-- <run_id>/
|       |-- figures/          # Privacy-utility plots
|       |-- tables/           # Summary, matrices, field leakage, significance tables
|       `-- run_manifest.json # Run configuration and metadata
|-- src/
|   |-- data_generation/      # Record formatting, synthetic fields, ground truth
|   |-- detection/            # Leakage and utility scoring
|   |-- evaluation/           # Experiment loop, aggregation, exports
|   |-- models/               # Ollama client
|   `-- prompting/            # Prompting strategy registry
`-- README.md
```

## Reproducibility

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Pull the local Ollama models:

```bash
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
```

Run the experiment:

```bash
python -m src.main
```

Generated outputs are saved under:

```text
results/runs/<run_id>/
```

Each run contains a manifest, summary metrics, field-level leakage breakdowns, comparison matrices, significance tables, and a privacy-utility trade-off scatter plot.

## Research Scope and Limitations

- Synthetic PII injection provides controlled evaluation targets, but it is not a complete substitute for naturally occurring privacy risk.
- Semantic leakage detection is auxiliary; the primary leakage evaluation is based on exact and partial matching of known sensitive fields.
- Utility scoring is based on rule-generated ground truth, which improves reproducibility but may not capture every useful human interpretation.
- Local model generation may vary between environments because generation is performed through Ollama and the experiment does not enforce a fixed model-side generation seed.

## Data Note

The source dataset is excluded from the repository for privacy reasons.
