"""
Utility Tools for the Portkey Class

This file module contains a collection of utility functions designed to enhance
the functionality and usability of the Portkey class
"""
from typing import Dict, Any, List
from enum import Enum
from llama_index.llms.base import LLMMetadata
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai_utils import (
    openai_modelname_to_contextsize,
    GPT3_5_MODELS,
    GPT4_MODELS,
    GPT3_MODELS,
    TURBO_MODELS,
    AZURE_TURBO_MODELS,
)
from llama_index.llms.anthropic_utils import CLAUDE_MODELS
from llama_index.llms.rubeus_utils import (
    ProviderTypes,
)
from .rubeus_utils import LLMBase, RubeusResponse

DEFAULT_MODEL = "gpt-3.5-turbo"

AVAILABLE_INTEGRATIONS = (OpenAI, Anthropic)

CLUADE_MODEL_FULLVERSION_MAP = {
    "claude-instant-1": "claude-instant-1.2",
    "claude-2": "claude-2.0",
}

ALL_AVAILABLE_MODELS = {
    **GPT4_MODELS,
    **TURBO_MODELS,
    **GPT3_5_MODELS,
    **GPT3_MODELS,
    **AZURE_TURBO_MODELS,
    **CLAUDE_MODELS,
}

CHAT_MODELS = {
    **GPT4_MODELS,
    **TURBO_MODELS,
    **AZURE_TURBO_MODELS,
}


def is_chat_model(model: str) -> bool:
    """Check if a given model is a chat-based language model.

    This function takes a model name or identifier as input and determines whether
    the model is designed for chat-based language generation, conversation, or
    interaction.

    Args:
        model (str): The name or identifier of the model to be checked.

    Returns:
        bool: True if the provided model is a chat-based language model, False otherwise.
    """
    return model in CHAT_MODELS


def generate_llm_metadata(llm: LLMBase) -> LLMMetadata:
    """
    Generate metadata for a Language Model (LLM) instance.

    This function takes an instance of a Language Model (LLM) and generates
    metadata based on the provided instance. The metadata includes information
    such as the context window, number of output tokens, chat model status,
    and model name.

    Parameters:
        llm (LLM): An instance of a Language Model (LLM) from which metadata
            will be generated.

    Returns:
        LLMMetadata: A data structure containing metadata attributes such as
            context window, number of output tokens, chat model status, and
            model name.

    Raises:
        ValueError: If the provided 'llm' is not an instance of llama_index.llms.base.LLM.
    """
    # if not isinstance(llm, LLM):
    #     raise ValueError("llm must be an instance of llama_index.llms.base.LLM")

    _num_output = None
    _context_window = None
    _is_chat_model = None
    _model_name = None
    if llm.provider == ProviderTypes.OPENAI:
        _context_window = openai_modelname_to_contextsize(llm.model)
        # _num_output = llm.max_tokens or -1
        # _is_chat_model = is_chat_model(llm.model)
        # _model_name = llm.model if hasattr(llm, "model") else None
    # elif isinstance(llm, Anthropic):
    #     _context_window = anthropic_modelname_to_contextsize(llm._model)
    #     _is_chat_model = is_chat_model(llm._model)
    #     _num_output = llm._max_tokens or -1
    #     _model_name = llm._model if hasattr(llm, "_model") else None

    return LLMMetadata(
        _context_window=_context_window,
        num_output=250,
        is_chat_model=True,
        model_name="gpt-3.5-turbo",
    )


def get_llm(response: RubeusResponse, llms: List[LLMBase]) -> LLMBase:
    fallback_llm = None
    for llm in llms:
        model = llm.model

        if model == response.model:
            fallback_llm = llm
            break
    if fallback_llm is None:
        raise ValueError("Failed to get the fallback LLM")
    return fallback_llm


class RubeusApiPaths(str, Enum):
    CHAT_COMPLETION = "/v1/chatComplete"
    COMPLETION = "/v1/complete"
