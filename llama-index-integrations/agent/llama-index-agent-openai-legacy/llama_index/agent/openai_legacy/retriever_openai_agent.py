"""Retriever OpenAI agent."""

import deprecated
from typing import Any, cast

from llama_index.agent.openai_legacy.openai_agent import (
    OpenAIAgent,
)
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.tools.types import BaseTool


@deprecated.deprecated(
    reason=(
        "FnRetrieverOpenAIAgent has been deprecated and is not maintained.\n\n"
        "`FunctionAgent` is the recommended replacement.\n\n"
        "See the docs for more information on updated agent usage: https://docs.llamaindex.ai/en/stable/understanding/agent/"
    ),
)
class FnRetrieverOpenAIAgent(OpenAIAgent):
    """
    Function Retriever OpenAI Agent.

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
