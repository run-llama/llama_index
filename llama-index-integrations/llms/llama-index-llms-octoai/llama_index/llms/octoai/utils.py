import logging
from typing import Dict, Sequence

from octoai.text_gen import ChatMessage as OctoAIChatMessage

from llama_index.core.base.llms.types import ChatMessage


TEXT_MODELS: Dict[str, int] = {
    "codellama-13b-instruct": 16384,
    "codellama-34b-instruct": 16384,
    "codellama-7b-instruct": 4096,
    "meta-llama-3-8b-instruct": 8192,
    "meta-llama-3-70b-instruct": 8192,
    "llama-2-13b-chat": 4096,
    "llama-2-70b-chat": 4096,
    "mistral-7b-instruct": 32768,
    "mixtral-8x7b-instruct": 32768,
    "mixtral-8x22b-instruct": 65536,
    "mixtral-8x22b-finetuned": 65536,
    "nous-hermes-2-mixtral-8x7b-dpo": 32768,
    "hermes-2-pro-mistral-7b": 32768,
    "llamaguard-7b": 4096,
    "qwen1.5-32b-chat": 32768,
}

ALL_AVAILABLE_MODELS = {**TEXT_MODELS}

MISSING_TOKEN_ERROR_MESSAGE = """No token found for OctoAI.
Please set the OCTOAI_TOKEN environment \
variable prior to initialization.
API keys can be found or created at \
https://octoai.cloud/settings
"""

logger = logging.getLogger(__name__)


def octoai_modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Examples:
        .. code-block:: python

            max_tokens = octoai.modelname_to_contextsize(TextModel.CODELLAMA_13B_INSTRUCT)
            max_tokens = octoai.modelname_to_contextsize("llama-2-13b-chat")
    """
    if modelname not in ALL_AVAILABLE_MODELS:
        print(
            "WARNING: Model not found in octoai.utils.py, returning a generous default value."
        )
        return 8192
    return ALL_AVAILABLE_MODELS[modelname]


def to_octoai_messages(messages: Sequence[ChatMessage]) -> Sequence[OctoAIChatMessage]:
    return [
        OctoAIChatMessage(content=message.content, role=message.role.value)
        for message in messages
    ]
