# Methodology

This project evaluates whether prompt engineering can reduce privacy leakage in LLM outputs generated from vehicular telematics records.

## Pipeline
1. Load telematics rows from CSV.
2. Convert each row into a prompting task.
3. Run five prompting strategies across the configured local Ollama models.
4. Detect exact and partial leakage by matching output text against sensitive values present in the input row.
5. Compute:
   - `leakage_score`
   - `normalized_leakage`
   - `exposure_rate`
   - `utility_score`
   - PUT index
6. Aggregate results by model and strategy.

## Experiment setup
- Dataset path: `data/tesla.csv`
- Task type: `concise_report`
- Sample size: 50 records
- Repetitions: 3
- Models: `llama3.1:8b`, `deepseek-r1:8b`

## Current prompting strategies
- no_instruction_control: control condition with no privacy instruction.
- direct_baseline: direct task prompt with brief privacy rules.
- policy_first_structured: explicit privacy policy before the task.
- least_to_most: decomposes the task into lower-risk operational observations before a final privacy review.
- skeleton_of_thought: uses a structured answer skeleton with an explicit privacy gate before final output.

## Core outputs
- `results/runs/<run_id>/run_manifest.json`
- `results/runs/<run_id>/tables/summary_metrics.csv`
- `results/runs/<run_id>/tables/field_leakage_breakdown.csv`
- `results/runs/<run_id>/tables/matrices/`
- `results/runs/<run_id>/tables/significance/`
- `results/runs/<run_id>/figures/put_scatter.png`
