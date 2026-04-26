# Privacy-Aware LLMs for Vehicular Data

## Overview
Research project evaluating whether prompt engineering can reduce sensitive information leakage in LLM outputs on vehicular telematics data.

## Method
- Local LLMs via Ollama (`llama3.1:8b`, `deepseek-r1:8b`)
- Prompting strategies:
  - `no_instruction_control`
  - `direct_baseline`
  - `policy_first_structured`
  - `least_to_most`
  - `skeleton_of_thought`
- Metrics:
  - `leakage_score`
  - `normalized_leakage`
  - `exposure_rate`
  - `utility_score`
  - PUT index

## Results
Generated results are stored under `results/runs/`, including summary tables, field-level leakage breakdowns, comparison matrices, significance tables, and a privacy-utility trade-off scatter plot.

## How to Run
pip install -r requirements.txt  
python -m src.main

## Note
Dataset excluded for privacy reasons.
