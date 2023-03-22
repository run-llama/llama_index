"""Wrapper functions around an LLM chain."""

# NOTE: moved to gpt_index/llm_predictor/base.py
# NOTE: this is for backwards compatibility

from gpt_index.llm_predictor.base import (  # noqa: F401
    LLMChain,
    LLMMetadata,
    LLMPredictor,
)
