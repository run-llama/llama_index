import asyncio
import json
import time
from abc import abstractmethod
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple, Type

from llama_index.agent.types import BaseAgent
from llama_index.callbacks.base import CallbackManager
from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.llms.base import LLM, ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_utils import is_function_calling_model
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.schema import BaseNode, NodeWithScore
from llama_index.tools import BaseTool, ToolOutput

DEFAULT_MAX_FUNCTION_CALLS = 5
DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"


class BaseOpenAIAgent(BaseAgent):
    def __init__(
        self,
        llm,
        memory,
        prefix_messages,
        verbose,
        max_function_calls,
        callback_manager,
        response_handler,
        stream_handler,
    ):
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.callback_manager = callback_manager or CallbackManager([])
        self.response_handler = response_handler
        self.stream_handler = stream_handler

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self._memory.get_all()

    def reset(self) -> None:
        self._memory.reset()

    def _get_latest_function_call(self) -> Optional[dict]:
        """Get latest function call from chat history."""
        return self._memory[-1].additional_kwargs.get("function_call", None)

    def _should_continue(n_function_calls):
        if n_function_calls >= self._max_function_calls:
            print(f"Exceeded max function calls: {self._max_function_calls}.")
            return False
        return True

    def init_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None):
        session = ChatSession(self._memory, self._prefix_messages, self._get_tools)
        tools, functions = session.init_chat(message, chat_history)
        latest_function_call = self._get_latest_function_call()
        return session, tools, functions, latest_function_call

    def handle_ai_response(self, session, tools, functions):
        all_messages = session.get_all_messages()
        chat_response = self.response_handler.get_response(all_messages, functions)
        return parse_response_and_call_function(
            chat_response, sessions, tools, functions
        )

    async def ahandle_ai_response(self, session, tools, functions):
        all_messages = session.get_all_messages()
        chat_response = await self.response_handler.get_response(
            all_messages, functions
        )
        return parse_response_and_call_function(
            chat_response, sessions, tools, functions
        )

    def parse_response_and_call_function(
        self, chat_response, sessions, tools, functions
    ):
        ai_message = chat_response.message
        session.memory.put(ai_message)

        function_call = session.get_latest_function_call()
        function_message, sources = None, []
        if function_call is not None:
            function_message, sources = call_function(
                tools, function_call, verbose=self._verbose
            )
            session.memory.put(function_message)

        return ai_message, function_call, sources

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        session, tools, functions, function_call = self.init_chat(message, chat_history)

        n_function_calls = 0
        while function_call is not None and self._should_continue(n_function_calls):
            ai_message, function_call, sources = self.handle_ai_response(
                session, tools, functions
            )
            n_function_calls += 1

        return AgentChatResponse(response=str(ai_message.content), sources=sources)

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        session, tools, functions, function_call = self.init_chat(message, chat_history)

        n_function_calls = 0
        while function_call is not None and self._should_continue(n_function_calls):
            ai_message, function_call, sources = await self.handle_ai_response(
                session, tools, functions
            )
            n_function_calls += 1

        return AgentChatResponse(response=str(ai_message.content), sources=sources)

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        session, tools, functions, function_call = self.init_chat(message, chat_history)
        chat_stream = self.stream_handler.start_stream(
            session.get_all_messages(), functions
        )

        n_function_calls = 0
        while function_call is not None and self._should_continue(n_function_calls):
            for chat_response in chat_stream:
                ai_message, function_call, sources = self.handle_ai_response(
                    session, tools, functions
                )
                n_function_calls += 1
                if function_call is None:
                    break

        return StreamingAgentChatResponse(
            response=str(ai_message.content), sources=sources
        )

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        session, tools, functions, function_call = self.init_chat(message, chat_history)
        chat_stream = await self.stream_handler.start_async_stream(
            session.get_all_messages(), functions
        )

        n_function_calls = 0
        function_call = None
        while function_call is not None and n_function_calls < self._max_function_calls:
            async for chat_response in chat_stream:
                ai_message, function_call, sources = await self.handle_ai_response(
                    session, tools, functions
                )
                n_function_calls += 1
                if function_call is None:
                    break

        return StreamingAgentChatResponse(
            response=str(ai_message.content), sources=sources
        )

    # ===== Query Engine Interface =====
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        agent_response = self.chat(
            query_bundle.query_str,
            chat_history=[],
        )
        return Response(response=str(agent_response))

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        agent_response = await self.achat(
            query_bundle.query_str,
            chat_history=[],
        )
        return Response(response=str(agent_response))


@dataclass
class ChatMessage:
    role: str
    content: str


class BaseMemory:
    def __init__(self):
        self.memory = []

    def get_all(self):
        return self.memory

    def put(self, message):
        self.memory.append(message)

    def reset(self):
        self.memory = []


class ChatSession:
    def __init__(
        self, memory: BaseMemory, prefix_messages: List[ChatMessage], get_tools_callback
    ):
        self.memory = memory
        self.prefix_messages = prefix_messages
        self.get_tools_callback = get_tools_callback
        self.tools = []
        self.functions = []

    def init_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Tuple[List[BaseTool], List[dict]]:
        if chat_history is not None:
            self.memory.set(chat_history)
        self.memory.put(ChatMessage(content=message, role=MessageRole.USER))
        self.tools = self.get_tools_callback(message)
        self.functions = [tool.metadata.to_openai_function() for tool in self.tools]
        return self.tools, self.functions

    def get_all_messages(self) -> List[ChatMessage]:
        return self.prefix_messages + self.memory.get()

    def get_latest_function_call(self) -> Optional[dict]:
        return self.memory.get_all()[-1].additional_kwargs.get("function_call", None)


class ChatResponseHandler:
    def __init__(self, llm):
        self.llm = llm

    def get_response(self, all_messages, functions):
        raise NotImplementedError


class SyncChatResponseHandler(ChatResponseHandler):
    def get_response(self, all_messages, functions):
        return self.llm.chat(all_messages, functions=functions)


class AsyncChatResponseHandler(ChatResponseHandler):
    async def get_response(self, all_messages, functions):
        return await self.llm.achat(all_messages, functions=functions)


class ChatStreamHandler:
    def __init__(self, llm):
        self.llm = llm

    def start_stream(self, all_messages, functions):
        return self.llm.stream_chat(all_messages, functions=functions)

    async def start_async_stream(self, all_messages, functions):
        return await self.llm.astream_chat(all_messages, functions=functions)


class ExperimentalOpenAIAgent(BaseOpenAIAgent):
    def __init__(
        self,
        tools: List[BaseTool],
        llm: OpenAI,
        memory: BaseMemory,
        response_handler,
        stream_handler,
        prefix_messages: List[ChatMessage],
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            memory=memory,
            response_handler=response_handler,
            stream_handler=stream_handler,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )
        self._tools = tools

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        response_handler=None,
        stream_handler=None,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> "OpenAIAgent":
        tools = tools or []
        chat_history = chat_history or []
        memory = memory or memory_cls.from_defaults(chat_history)
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")

        if not is_function_calling_model(llm.model):
            raise ValueError(
                f"Model name {llm.model} does not support function calling API. "
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
            llm=llm,
            memory=memory,
            response_handler=response_handler,
            stream_handler=stream_handler,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )

    def _get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._tools


class RetrieverOpenAIAgent(BaseOpenAIAgent):
    """Retriever OpenAI Agent.

    This agent specifically performs retrieval on top of functions
    during query-time.

    NOTE: this is a beta feature, function interfaces might change.
    NOTE: this is also a too generally named, a better name is
        FunctionRetrieverOpenAIAgent

    TODO: add a native OpenAI Tool Index.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        node_to_tool_fn: Callable[[BaseNode], BaseTool],
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
        self._retriever = retriever
        self._node_to_tool_fn = node_to_tool_fn

    @classmethod
    def from_retriever(
        cls,
        retriever: BaseRetriever,
        node_to_tool_fn: Callable[[BaseNode], BaseTool],
        llm: Optional[OpenAI] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
    ) -> "RetrieverOpenAIAgent":
        chat_history = chat_history or []
        memory = memory or memory_cls.from_defaults(chat_history)

        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")

        if not is_function_calling_model(llm.model):
            raise ValueError(
                f"Model name {llm.model} does not support function calling API. "
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
            node_to_tool_fn=node_to_tool_fn,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )

    def _get_tools(self, message: str) -> List[BaseTool]:
        retrieved_nodes_w_scores: List[NodeWithScore] = self._retriever.retrieve(
            message
        )
        retrieved_nodes = [node.node for node in retrieved_nodes_w_scores]
        retrieved_tools: List[BaseTool] = [
            self._node_to_tool_fn(n) for n in retrieved_nodes
        ]
        return retrieved_tools
