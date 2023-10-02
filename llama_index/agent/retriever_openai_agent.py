"""Retriever OpenAI agent."""

from typing import Any, cast

from llama_index.agent.openai_agent import (
    OpenAIAgent,
)
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
        return cast(
            FnRetrieverOpenAIAgent, cls.from_tools(tool_retriever=retriever, **kwargs)
        )
