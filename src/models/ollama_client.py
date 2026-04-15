from __future__ import annotations

import requests

from src.config import ModelConfig, load_json, CONFIG_DIR


class OllamaClient:
    def __init__(self, url: str | None = None) -> None:
        if url is None:
            raw = load_json(CONFIG_DIR / "model_config.json")
            url = raw["ollama_url"]
        self.url = url

    def generate(self, prompt: str, model: ModelConfig) -> str:
        payload = {
            "model": model.name,
            "prompt": prompt,
            "stream": False,
            "temperature": model.temperature,
            "num_ctx": model.num_ctx,
            "num_predict": model.num_predict,
        }

        response = requests.post(self.url, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()

        if "response" not in data:
            raise ValueError(f"Unexpected Ollama response: {data}")

        return str(data["response"]).strip()