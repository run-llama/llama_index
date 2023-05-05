"""Init params."""

# TODO: move LLMPredictor to this folder
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.llm_predictor.stable_lm import StableLMPredictor
from llama_index.llm_predictor.structured import StructuredLLMPredictor

__all__ = ["LLMPredictor", "StructuredLLMPredictor", "StableLMPredictor"]
