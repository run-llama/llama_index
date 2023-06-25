"""Retriever OpenAI agent."""

from llama_index.agent.openai_agent import BaseOpenAIAgent
from llama_index.objects.base import ObjectRetriever
from llama_index.tools.types import BaseTool
from typing import Optional, List
from llama_index.bridge.langchain import ChatOpenAI, ChatMessageHistory
from llama_index.callbacks.base import CallbackManager
from llama_index.agent.openai_agent import (
    SUPPORTED_MODEL_NAMES,
    DEFAULT_MAX_FUNCTION_CALLS,
)


class FnRetrieverOpenAIAgent(BaseOpenAIAgent):
    """Function Retriever OpenAI Agent.

    Uses our object retriever module to retrieve openai agent.

    """

    def __init__(
        self,
        retriever: ObjectRetriever[BaseTool],
        llm: ChatOpenAI,
        chat_history: ChatMessageHistory,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            chat_history=chat_history,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )
        self._retriever = retriever

    @classmethod
    def from_retriever(
        cls,
        retriever: ObjectRetriever[BaseTool],
        llm: Optional[ChatOpenAI] = None,
        chat_history: Optional[ChatMessageHistory] = None,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "FnRetrieverOpenAIAgent":
        lc_chat_history = chat_history or ChatMessageHistory()
        llm = llm or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
        if not isinstance(llm, ChatOpenAI):
            raise ValueError("llm must be a ChatOpenAI instance")

        if llm.model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Model name {llm.model_name} not supported. "
                f"Supported model names: {SUPPORTED_MODEL_NAMES}"
            )

        return cls(
            retriever=retriever,
            llm=llm,
            chat_history=lc_chat_history,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )

    def _get_tools(self, message: str) -> List[BaseTool]:
        tools = self._retriever.retrieve(message)
        return tools
