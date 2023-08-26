"""
Utility Tools for the Portkey Class

This file module contains a collection of utility functions designed to enhance
the functionality and usability of the Portkey class
"""
from typing import Sequence, Dict, Any, Optional
from enum import Enum
from llama_index.llms.base import LLMMetadata, LLM, ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai_utils import (
    openai_modelname_to_contextsize,
    GPT3_5_MODELS,
    GPT4_MODELS,
    GPT3_MODELS,
    TURBO_MODELS,
    AZURE_TURBO_MODELS,
    DISCONTINUED_MODELS as OPENAI_DISCONTINUED_MODELS
)
from llama_index.llms.anthropic_utils import CLAUDE_MODELS
from llama_index.llms.anthropic import anthropic_modelname_to_contextsize
from llama_index.llms.rubeus_utils import ProviderTypes, RubeusCacheType


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


class PortkeyParams(Dict[str, Any]):
    """
    provider (Optional[ProviderTypes]): The LLM provider to be used for the Portkey integration.
        Eg: openai, anthropic etc.
        NOTE: Check the ProviderTypes to see the supported list of LLMs.
    model (str): The name of the language model to use (default: "gpt-3.5-turbo").
    temperature (float): The temperature parameter for text generation (default: 0.1).
    max_tokens (Optional[int]): The maximum number of tokens in the generated text.
    max_retries (int): The maximum number of retries for failed requests (default: 5).
    trace_id (Optional[str]): A unique identifier for tracing requests.
    cache_status (Optional[RubeusCacheType]): The type of cache to use (default: "").
        If cache_status is set, then cache is automatically set to True
    cache (Optional[bool]): Whether to use caching (default: False).
    metadata (Optional[Dict[str, Any]]): Metadata associated with the request (default: {}).
    weight (Optional[float]): The weight of the LLM in the ensemble (default: 1.0).
    """

    provider: Optional[ProviderTypes]
    model: str
    model_api_key: str
    temperature: float
    max_tokens: Optional[int]
    max_retries: int
    trace_id: Optional[str]
    cache_status: Optional[RubeusCacheType]
    cache: Optional[bool]
    metadata: Dict[str, Any]
    weight: Optional[float]
    prompt: Optional[str]
    messages: Optional[ChatMessage]


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


def generate_llm_metadata(llm: Optional[LLM]) -> LLMMetadata:
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
    # if isinstance(llm, OpenAI):
    #     _context_window = openai_modelname_to_contextsize(llm.model)
    #     _num_output = llm.max_tokens or -1
    #     _is_chat_model = is_chat_model(llm.model)
    #     _model_name = llm.model if hasattr(llm, "model") else None
    # elif isinstance(llm, Anthropic):
    #     _context_window = anthropic_modelname_to_contextsize(llm._model)
    #     _is_chat_model = is_chat_model(llm._model)
    #     _num_output = llm._max_tokens or -1
    #     _model_name = llm._model if hasattr(llm, "_model") else None

    return LLMMetadata(
        context_window=1000,
        num_output=250,
        is_chat_model=True,
        model_name="gpt-3.5-turbo",
    )


def get_fallback_llm(response: Dict[str, Any], llms: Sequence[LLM]) -> LLM:
    response_model = response["model"]
    fallback_llm = None
    for llm in llms:
        model = None
        if isinstance(llm, OpenAI):
            model = llm.model if hasattr(llm, "model") else None
        elif isinstance(llm, Anthropic):
            name = llm._model if hasattr(llm, "_model") else None
            model = CLUADE_MODEL_FULLVERSION_MAP[name] if name is not None else None

        if model == response_model:
            fallback_llm = llm
            break

    if fallback_llm is None:
        raise ValueError(
            f"Response model '{response_model}' name doesn't match the input model names. Please verify the fallback model names."
        )

    return fallback_llm


class RubeusApiPaths(str, Enum):
    CHAT_COMPLETION = "/v1/chatComplete"
    COMPLETION = "/v1/complete"
