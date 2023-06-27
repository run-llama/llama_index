"""Init params."""

from llama_index.llm_predictor.base import LLMPredictor

# NOTE: this results in a circular import
# from llama_index.llm_predictor.mock import MockLLMPredictor
from llama_index.llm_predictor.structured import StructuredLLMPredictor

__all__ = [
    "LLMPredictor",
    # NOTE: this results in a circular import
    # "MockLLMPredictor",
    "StructuredLLMPredictor",
]
