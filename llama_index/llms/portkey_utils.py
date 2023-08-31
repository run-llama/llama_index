"""
Utility Tools for the Portkey Class

This file module contains a collection of utility functions designed to enhance
the functionality and usability of the Portkey class
"""
from typing import List
from enum import Enum
from rubeus import LLMBase, RubeusResponse
from llama_index.llms.base import LLMMetadata
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai_utils import (
    GPT3_5_MODELS,
    GPT4_MODELS,
    GPT3_MODELS,
    TURBO_MODELS,
    AZURE_TURBO_MODELS,
)
from llama_index.llms.anthropic_utils import CLAUDE_MODELS


DISCONTINUED_MODELS = {
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}

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
        bool: True if the provided model is a chat-based language model,
        False otherwise.
    """
    return model in CHAT_MODELS


def modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = modelname_to_contextsize("text-davinci-003")
    """
    # handling finetuned models
    if "ft-" in modelname:  # legacy fine-tuning
        modelname = modelname.split(":")[0]
    elif modelname.startswith("ft:"):
        modelname = modelname.split(":")[1]

    if modelname in DISCONTINUED_MODELS:
        raise ValueError(
            f"Model {modelname} has been discontinued. " "Please choose another model."
        )

    context_size = ALL_AVAILABLE_MODELS.get(modelname, None)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid model name."
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


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
        ValueError: If the provided 'llm' is not an instance of
        llama_index.llms.base.LLM.
    """
    if not isinstance(llm, LLMBase):
        raise ValueError("llm must be an instance of rubeus.LLMBase")

    return LLMMetadata(
        _context_window=modelname_to_contextsize(llm.model),
        is_chat_model=is_chat_model(llm.model),
        model_name=llm.model,
    )


def get_llm(response: RubeusResponse, llms: List[LLMBase]) -> LLMBase:
    # TODO: Update this logic over here.
    fallback_llm = LLMBase.construct()
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
