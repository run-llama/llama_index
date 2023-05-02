"""Init params."""

# TODO: move LLMPredictor to this folder
from gpt_index.llm_predictor.base import LLMPredictor
from gpt_index.llm_predictor.stable_lm import StableLMPredictor
from gpt_index.llm_predictor.structured import StructuredLLMPredictor
from gpt_index.llm_predictor.ai21_contextual import AI21ContextualAnswersPredictor

__all__ = [
    "LLMPredictor",
    "StructuredLLMPredictor",
    "StableLMPredictor",
    "AI21ContextualAnswersPredictor",
]
