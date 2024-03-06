"""PremAI's API to interact with deployed projects"""

import os
import typing
from typing import Any, Dict, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

if typing.TYPE_CHECKING:
    import premai as premai


class PremAI(CustomLLM):
    """PremAI LLM Provider"""

    project_id: int = Field(
        description=(
            "The project ID in which the experiments or deployements are carried out. can find all your projects here: https://app.premai.io/projects/"
        )
    )

    model_name: Optional[str] = Field(
        default=None,
        description=(
            "Name of the model. This is an optional paramter. The default model is the one deployed from Prem's LaunchPad. An example: https://app.premai.io/projects/<project-id>/launchpad. If model name is other than default model then it will override the calls from the model deployed from launchpad."
        ),
    )

    _client: "premai.Prem" = PrivateAttr()
