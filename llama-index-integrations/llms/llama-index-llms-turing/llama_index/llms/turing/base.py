"""
LlamaIndex LLM Adapter for Turing Engine.
Enables RAG query engines, indexes, and retrievers backed by Turing Engine.
"""

from typing import Any, Dict, List, Optional
import json
import urllib.request

class TuringEngineLLM:
    """
    TuringEngineLLM adapter for LlamaIndex query pipelines.
    """
    def __init__(
        self,
        model: str = "llama-3.1-70b",
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "turing-local",
        temperature: float = 0.7,
        max_tokens: int = 256
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str, **kwargs) -> Any:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False
        }

        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = data["choices"][0]["message"]["content"]
        
        class CompletionResponse:
            def __init__(self, t):
                self.text = t
            def __str__(self):
                return self.text

        return CompletionResponse(text)
