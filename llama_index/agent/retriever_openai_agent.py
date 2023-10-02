"""Retriever OpenAI agent."""

from typing import List, Optional, Type, Any

from llama_index.agent.openai_agent import (
    DEFAULT_MAX_FUNCTION_CALLS,
    DEFAULT_MODEL_NAME,
    BaseOpenAIAgent,
    OpenAIAgent,
)
from llama_index.callbacks.base import CallbackManager
from llama_index.llms.base import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_utils import is_function_calling_model
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.objects.base import ObjectRetriever
from llama_index.tools.types import BaseTool


class FnRetrieverOpenAIAgent(OpenAIAgent):
    """Function Retriever OpenAI Agent.

    Uses our object retriever module to retrieve openai agent.

    NOTE: This is deprecated, you can just use the base `OpenAIAgent` class by
    specifying the following:
    ```
    agent = OpenAIAgent.from_tools(tool_retriever=retriever, ...)
    ```

    """

    @classmethod
    def from_retriever(
        cls, retriever: ObjectRetriever[BaseTool], **kwargs: Any
    ) -> "FnRetrieverOpenAIAgent":
        return cls.from_tools(tool_retriever=retriever, **kwargs)
