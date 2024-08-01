"""Init params."""

from llama_index.legacy.llm_predictor.base import LLMPredictor

# NOTE: this results in a circular import
# from llama_index.legacy.llm_predictor.mock import MockLLMPredictor
from llama_index.legacy.llm_predictor.structured import StructuredLLMPredictor

__all__ = [
    "LLMPredictor",
    # NOTE: this results in a circular import
    # "MockLLMPredictor",
    "StructuredLLMPredictor",
]
