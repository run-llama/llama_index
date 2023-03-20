"""Init params."""

# TODO: move LLMPredictor to this folder
from gpt_index.llm_predictor.base import LLMPredictor
from gpt_index.llm_predictor.structured import StructuredLLMPredictor

__all__ = [
    "LLMPredictor",
    "StructuredLLMPredictor",
]
