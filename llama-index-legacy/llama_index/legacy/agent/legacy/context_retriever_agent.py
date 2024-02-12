"""Context retriever agent."""

from typing import List, Optional, Type, Union

from llama_index.legacy.agent.legacy.openai_agent import (
    DEFAULT_MAX_FUNCTION_CALLS,
    DEFAULT_MODEL_NAME,
    BaseOpenAIAgent,
)
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.chat_engine.types import (
    AgentChatResponse,
)
from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.legacy.core.llms.types import ChatMessage
from llama_index.legacy.llms.llm import LLM
from llama_index.legacy.llms.openai import OpenAI
from llama_index.legacy.llms.openai_utils import is_function_calling_model
from llama_index.legacy.memory import BaseMemory, ChatMemoryBuffer
from llama_index.legacy.prompts import PromptTemplate
from llama_index.legacy.schema import NodeWithScore
from llama_index.legacy.tools import BaseTool
from llama_index.legacy.utils import print_text

# inspired by DEFAULT_QA_PROMPT_TMPL from llama_index.legacy/prompts/default_prompts.py
DEFAULT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "either pick the corresponding tool or answer the function: {query_str}\n"
)
DEFAULT_QA_PROMPT = PromptTemplate(DEFAULT_QA_PROMPT_TMPL)


class ContextRetrieverOpenAIAgent(BaseOpenAIAgent):
    """ContextRetriever OpenAI Agent.

    This agent performs retrieval from BaseRetriever before
    calling the LLM. Allows it to augment user message with context.

    NOTE: this is a beta feature, function interfaces might change.

    Args:
        tools (List[BaseTool]): A list of tools.
        retriever (BaseRetriever): A retriever.
        qa_prompt (Optional[PromptTemplate]): A QA prompt.
        context_separator (str): A context separator.
        llm (Optional[OpenAI]): An OpenAI LLM.
        chat_history (Optional[List[ChatMessage]]): A chat history.
        prefix_messages: List[ChatMessage]: A list of prefix messages.
        verbose (bool): Whether to print debug statements.
        max_function_calls (int): Maximum number of function calls.
        callback_manager (Optional[CallbackManager]): A callback manager.

    """

    def __init__(
        self,
        tools: List[BaseTool],
        retriever: BaseRetriever,
        qa_prompt: PromptTemplate,
        context_separator: str,
        llm: OpenAI,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )
        self._tools = tools
        self._qa_prompt = qa_prompt
        self._retriever = retriever
        self._context_separator = context_separator

    @classmethod
    def from_tools_and_retriever(
        cls,
        tools: List[BaseTool],
        retriever: BaseRetriever,
        qa_prompt: Optional[PromptTemplate] = None,
        context_separator: str = "\n",
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
    ) -> "ContextRetrieverOpenAIAgent":
        """Create a ContextRetrieverOpenAIAgent from a retriever.

        Args:
            retriever (BaseRetriever): A retriever.
            qa_prompt (Optional[PromptTemplate]): A QA prompt.
            context_separator (str): A context separator.
            llm (Optional[OpenAI]): An OpenAI LLM.
            chat_history (Optional[ChatMessageHistory]): A chat history.
            verbose (bool): Whether to print debug statements.
            max_function_calls (int): Maximum number of function calls.
            callback_manager (Optional[CallbackManager]): A callback manager.

        """
        qa_prompt = qa_prompt or DEFAULT_QA_PROMPT
        chat_history = chat_history or []
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")
        if callback_manager is not None:
            llm.callback_manager = callback_manager

        memory = memory or memory_cls.from_defaults(chat_history=chat_history, llm=llm)

        if not is_function_calling_model(llm.model):
            raise ValueError(
                f"Model name {llm.model} does not support function calling API."
            )
        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        return cls(
            tools=tools,
            retriever=retriever,
            qa_prompt=qa_prompt,
            context_separator=context_separator,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )

    def _get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._tools

    def _build_formatted_message(self, message: str) -> str:
        # augment user message
        retrieved_nodes_w_scores: List[NodeWithScore] = self._retriever.retrieve(
            message
        )
        retrieved_nodes = [node.node for node in retrieved_nodes_w_scores]
        retrieved_texts = [node.get_content() for node in retrieved_nodes]

        # format message
        context_str = self._context_separator.join(retrieved_texts)
        return self._qa_prompt.format(context_str=context_str, query_str=message)

    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        """Chat."""
        formatted_message = self._build_formatted_message(message)
        if self._verbose:
            print_text(formatted_message + "\n", color="yellow")

        return super().chat(
            formatted_message, chat_history=chat_history, tool_choice=tool_choice
        )

    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        """Chat."""
        formatted_message = self._build_formatted_message(message)
        if self._verbose:
            print_text(formatted_message + "\n", color="yellow")

        return await super().achat(
            formatted_message, chat_history=chat_history, tool_choice=tool_choice
        )

    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._get_tools(message)
