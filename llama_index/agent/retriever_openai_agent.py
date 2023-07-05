"""Retriever OpenAI agent."""

from typing import List, Optional

from llama_index.agent.openai_agent import (DEFAULT_MAX_FUNCTION_CALLS,
                                            DEFAULT_MODEL_NAME,
                                            SUPPORTED_MODEL_NAMES,
                                            BaseOpenAIAgent)
from llama_index.callbacks.base import CallbackManager
from llama_index.llms.base import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.objects.base import ObjectRetriever
from llama_index.tools.types import BaseTool


class FnRetrieverOpenAIAgent(BaseOpenAIAgent):
    """Function Retriever OpenAI Agent.

    Uses our object retriever module to retrieve openai agent.

    """

    def __init__(
        self,
        retriever: ObjectRetriever[BaseTool],
        llm: OpenAI,
        chat_history: List[ChatMessage],
        prefix_messages: List[ChatMessage],
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            chat_history=chat_history,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )
        self._retriever = retriever

    @classmethod
    def from_retriever(
        cls,
        retriever: ObjectRetriever[BaseTool],
        llm: Optional[OpenAI] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
    ) -> "FnRetrieverOpenAIAgent":
        chat_history = chat_history or []
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")

        if llm.model not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Model name {llm.model} not supported. "
                f"Supported model names: {SUPPORTED_MODEL_NAMES}"
            )
        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        return cls(
            retriever=retriever,
            llm=llm,
            chat_history=chat_history,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )

    def _get_tools(self, message: str) -> List[BaseTool]:
        tools = self._retriever.retrieve(message)
        return tools
