"""Function calling agent."""

from typing import Any, List, Optional

from llama_index.core.agent.runner.base import AgentRunner, AgentState
from llama_index.core.agent.function_calling.step import (
    FunctionCallingAgentWorker,
    DEFAULT_MAX_FUNCTION_CALLS,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.tools.types import BaseTool


class FunctionCallingAgent(AgentRunner):
    """Function calling agent.

    Light wrapper around AgentRunner.
    """

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[FunctionCallingLLM] = None,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        state: Optional[AgentState] = None,
        allow_parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> "FunctionCallingAgent":
        """Create a FunctionCallingAgent from a list of tools."""
        tools = tools or []

        llm = llm or Settings.llm  # type: ignore
        assert isinstance(
            llm, FunctionCallingLLM
        ), "llm must be an instance of FunctionCallingLLM"

        if callback_manager is not None:
            llm.callback_manager = callback_manager

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tools,
            tool_retriever=tool_retriever,
            llm=llm,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            prefix_messages=prefix_messages,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
        )

        return cls(
            agent_worker=agent_worker,
            memory=memory,
            chat_history=chat_history,
            state=state,
            llm=llm,
            callback_manager=callback_manager,
            verbose=verbose,
            **kwargs,
        )
