# Privacy-Aware LLMs for Vehicular Data

## Overview
Research project evaluating whether prompt engineering can reduce sensitive information leakage in LLM outputs on vehicular telematics data.

## Method
- Local LLMs via Ollama (LLaMA 3, DeepSeek)
- Prompting strategies: baseline, policy-first, least-to-most, skeleton-of-thought
- Metrics: leakage score, exposure rate, utility score

## Results
(Will include tables/plots comparing privacy–utility trade-offs)

## How to Run
pip install -r requirements.txt  
python -m src.main

## Note
Dataset excluded for privacy reasons.
