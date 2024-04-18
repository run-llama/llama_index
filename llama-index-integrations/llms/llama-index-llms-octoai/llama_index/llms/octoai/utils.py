import logging
from typing import Dict, Sequence
from llama_index.core.base.llms.types import ChatMessage
from octoai.chat import TextModel

TEXT_MODELS: Dict[str, int] = {
    TextModel.CODELLAMA_13B_INSTRUCT: 16384,
    TextModel.CODELLAMA_34B_INSTRUCT: 16384,
    TextModel.CODELLAMA_70B_INSTRUCT: 16384,
    TextModel.CODELLAMA_7B_INSTRUCT: 4096,
    TextModel.LLAMA_2_13B_CHAT: 4096,
    TextModel.LLAMA_2_70B_CHAT: 4096,
    TextModel.MISTRAL_7B_INSTRUCT: 32768,
    TextModel.MIXTRAL_8X7B_INSTRUCT: 32768,
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
        raise ValueError(
            f"Unknown model {modelname!r}. Please provide a supported model name as \
            a string or using the TextModel enum from the OctoAI SDK:"
            f" {', '.join(ALL_AVAILABLE_MODELS.keys())}"
        )
    return ALL_AVAILABLE_MODELS[modelname]


def to_octoai_messages(messages: Sequence[ChatMessage]) -> Sequence[Dict]:
    return [
        {"role": message.role.value, "content": message.content} for message in messages
    ]
