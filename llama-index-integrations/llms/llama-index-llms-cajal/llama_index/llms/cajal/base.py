from typing import Any, Optional, List, Mapping
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
import requests
import json


class CajalLLM(CustomLLM):
    """CAJAL LLM integration for LlamaIndex.
    
    A fine-tuned 4B model for generating scientific papers with real arXiv citations.
    Runs locally via Ollama, vLLM, or llama.cpp.
    
    Example:
        llm = CajalLLM(base_url="http://localhost:11434", model="cajal-p2pclaw")
        response = llm.complete("Generate a paper on quantum machine learning")
    """

    base_url: str = "http://localhost:11434"
    model: str = "cajal-p2pclaw"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = (
        "You are CAJAL, a scientific paper generator. "
        "Generate 7-section papers with real arXiv citations. "
        "Structure: Abstract, Introduction, Methodology, Results, Discussion, Conclusion, References."
    )

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "cajal-p2pclaw", **kwargs: Any):
        super().__init__(base_url=base_url, model=model, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=32768,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Generate a completion using Ollama API."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return CompletionResponse(text=data.get("response", ""))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream a completion using Ollama API."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        response = requests.post(url, json=payload, stream=True, timeout=300)
        response.raise_for_status()
        
        accumulated = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                chunk = data.get("response", "")
                accumulated += chunk
                yield CompletionResponse(text=accumulated, delta=chunk)

    @property
    def _llm_type(self) -> str:
        return "cajal"


def generate_scientific_paper(
    topic: str,
    llm: Optional[CajalLLM] = None,
    include_tribunal: bool = True,
) -> str:
    """High-level helper to generate a full scientific paper with optional tribunal scoring.
    
    Args:
        topic: Research topic for the paper
        llm: CajalLLM instance (creates default if None)
        include_tribunal: Whether to run tribunal scoring
        
    Returns:
        Complete paper text with tribunal report if enabled
    """
    if llm is None:
        llm = CajalLLM()
    
    prompt = (
        f"Generate a complete 7-section scientific paper on: {topic}\n\n"
        "Sections required:\n"
        "1. Abstract (150 words)\n"
        "2. Introduction (500 words)\n"
        "3. Methodology (400 words)\n"
        "4. Results (400 words)\n"
        "5. Discussion (400 words)\n"
        "6. Conclusion (200 words)\n"
        "7. References (BibTeX format, verified arXiv citations)\n\n"
        "Include real arXiv citations for every reference."
    )
    
    response = llm.complete(prompt)
    paper = response.text
    
    if include_tribunal:
        # Tribunal scoring simulation
        tribunal_prompt = (
            f"Review the following scientific paper and score each section (0-10):\n\n{paper}\n\n"
            "Provide scores for: Scientific Rigor, Clarity, Novelty, Citation Quality. "
            "List sections scoring below 7.0 that need revision."
        )
        tribunal_response = llm.complete(tribunal_prompt)
        paper += f"\n\n---\n\n## Tribunal Report\n\n{tribunal_response.text}"
    
    return paper
