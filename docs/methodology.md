# Methodology

This project evaluates whether prompt engineering can reduce privacy leakage in LLM outputs generated from vehicular telematics records.

## Pipeline
1. Load telematics rows from CSV.
2. Convert each row into a prompting task.
3. Run multiple prompting strategies across one or more local Ollama models.
4. Detect exact and partial leakage by matching output text against sensitive values present in the input row.
5. Compute:
   - leakage score
   - exposure rate
   - utility score
6. Aggregate results by model and strategy.

## Current prompting strategies
- no_instruction_control: control condition with no privacy instruction.
- direct_baseline: direct task prompt with brief privacy rules.
- policy_first_structured: explicit privacy policy before the task.
- least_to_most: decomposes the task into lower-risk operational observations before a final privacy review.
- skeleton_of_thought: uses a structured answer skeleton with an explicit privacy gate before final output.

## Core outputs
- `results/raw/results.csv`
- `results/tables/summary_metrics.csv`
