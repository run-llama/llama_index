"""ReAct agent.

Simple wrapper around AgentEngine + ReActAgentStepEngine.

For the legacy implementation see:
```python
from llama_index.agent.legacy.react.base import ReActAgent
```

"""
from llama_index.agent.legacy.react.base import ReActAgent

from llama_index.agent.types import BaseAgent
from llama_index.agent.react.step import ReActAgentStepEngine
from llama_index.agent.executor.base import AgentEngine

import asyncio
from itertools import chain
from threading import Thread
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    cast,
)

from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.agent.types import BaseAgent
from llama_index.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.llms.base import LLM, ChatMessage, ChatResponse, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.memory.types import BaseMemory
from llama_index.objects.base import ObjectRetriever
from llama_index.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.tools.types import AsyncBaseTool
from llama_index.utils import print_text, unit_generator


DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"

class ReActAgent(BaseAgent):
    """ReAct agent.

    Simple wrapper around AgentEngine + ReActAgentStepEngine.

    For the legacy implementation see:
    ```python
    from llama_index.agent.legacy.react.base import ReActAgent
    ```
    
    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        memory: BaseMemory,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
    ) -> None:
        super().__init__(callback_manager=callback_manager or llm.callback_manager)

        self._step_engine = ReActAgentStepEngine.from_tools(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=self.callback_manager,
            verbose=verbose,
        )
        self._agent_engine = AgentEngine(
            self._step_engine,
            memory=memory,
            llm=llm,
            callback_manager=self.callback_manager,
        )

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "ReActAgent":
        """Convenience constructor method from set of of BaseTools (Optional).

        NOTE: kwargs should have been exhausted by this point. In other words
        the various upstream components such as BaseSynthesizer (response synthesizer)
        or BaseRetriever should have picked up off their respective kwargs in their
        constructions.

        Returns:
            ReActAgent
        """
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        memory = memory or memory_cls.from_defaults(
            chat_history=chat_history or [], llm=llm
        )
        return cls(
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            memory=memory,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
        )
    
    @property
    def chat_history(self) -> List[ChatMessage]:
        """Chat history."""
        return self._agent_engine.memory.get_all()

    def reset(self) -> None:
        self._agent_engine.memory.reset()

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Chat."""
        return self._agent_engine.chat(message=message, chat_history=chat_history)
    
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Chat."""
        return await self._agent_engine.achat(message=message, chat_history=chat_history)

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """Stream chat."""
        return self._agent_engine.stream_chat(
            message=message, chat_history=chat_history
        )

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """Async stream chat."""
        return await self._agent_engine.astream_chat(
            message=message, chat_history=chat_history
        )

