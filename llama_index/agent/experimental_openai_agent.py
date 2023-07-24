import asyncio
import json
import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple, Type

from llama_index.agent.types import BaseAgent
from llama_index.callbacks.base import CallbackManager
from llama_index.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.llms.base import LLM, ChatMessage, ChatResponse, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_utils import is_function_calling_model
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.schema import BaseNode, NodeWithScore
from llama_index.tools import BaseTool, ToolOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

DEFAULT_MAX_FUNCTION_CALLS = 5
DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"


def get_function_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    """Get function by name."""
    name_to_tool = {tool.metadata.name: tool for tool in tools}
    if name not in name_to_tool:
        raise ValueError(f"Tool with name {name} not found")
    return name_to_tool[name]


def call_function(
    tools: List[BaseTool], function_call: dict, verbose: bool = False
) -> Tuple[ChatMessage, ToolOutput]:
    """Call a function and return the output as a string."""
    name = function_call["name"]
    arguments_str = function_call["arguments"]
    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = get_function_by_name(tools, name)
    argument_dict = json.loads(arguments_str)
    output = tool(**argument_dict)
    if verbose:
        print(f"Got output: {str(output)}")
        print("========================")
    return (
        ChatMessage(
            content=str(output),
            role=MessageRole.FUNCTION,
            additional_kwargs={
                "name": function_call["name"],
            },
        ),
        output,
    )


class ChatMode(Enum):
    default = 0
    stream = 1


class ChatSession:
    def __init__(
        self, memory: BaseMemory, prefix_messages: List[ChatMessage], get_tools_callback
    ):
        self.memory = memory
        self.prefix_messages = prefix_messages
        self.get_tools_callback = get_tools_callback
        self.tools = []
        self.functions = []

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self.memory.get_all()

    def reset(self) -> None:
        self.memory.reset()

    def prepare_message(self, message: str) -> Tuple[List[BaseTool], List[dict]]:
        """Prepare tools and functions for the message."""
        self.tools = self.get_tools_callback(message)
        self.functions = [tool.metadata.to_openai_function() for tool in self.tools]
        return self.tools, self.functions

    def get_all_messages(self) -> List[ChatMessage]:
        return self.prefix_messages + self.memory.get()

    def get_latest_function_call(self) -> Optional[dict]:
        return self.memory.get_all()[-1].additional_kwargs.get("function_call", None)


class BaseOpenAIAgent(BaseAgent):
    def __init__(
        self,
        llm: OpenAI,
        memory,
        prefix_messages,
        verbose,
        max_function_calls,
        callback_manager,
    ):
        self._llm = llm
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.callback_manager = callback_manager or CallbackManager([])
        self.session = ChatSession(memory, prefix_messages, self._get_tools)
        self.sources = []

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self.session.chat_history

    @property
    def all_messages(self):
        return self.session.get_all_messages()

    def reset(self) -> None:
        self.session.reset()

    def _should_continue(self, function_call, n_function_calls):
        if n_function_calls > self._max_function_calls:
            return False
        if not function_call:
            return False
        return True

    def init_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None):
        if chat_history is not None:
            self.session.memory.set(chat_history)
        self.sources = []
        self.session.memory.put(ChatMessage(content=message, role=MessageRole.USER))
        tools, functions = self.session.prepare_message(message)
        return tools, functions

    def _process_message(self, chat_response: ChatResponse) -> AgentChatResponse:
        ai_message = chat_response.message
        self.session.memory.put(ai_message)
        return AgentChatResponse(response=str(ai_message.content), sources=self.sources)

    def _get_stream_ai_response(self, functions) -> StreamingAgentChatResponse:
        chat_stream_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(self.all_messages, functions=functions),
            sources=self.sources,
        )

        # Get the response in a separate thread so we can yield the response
        thread = Thread(
            target=chat_stream_response.write_response_to_history,
            args=(self.session.memory,),
        )
        logger.debug("Start thread")
        thread.start()

        while chat_stream_response._is_function is None:
            # Wait until we know if the response is a function call or not
            time.sleep(0.05)
            if chat_stream_response._is_function is False:
                logger.debug("return stream")
                return chat_stream_response
        else:
            logger.debug("chat_stream_response._is_function is not None")

        logger.debug("Join thread")
        thread.join()
        logger.debug("Thread joined, rtn response")
        return chat_stream_response

    async def _get_async_stream_ai_response(self, functions):
        chat_stream_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(
                self.all_messages, functions=functions
            ),
            sources=self.sources,
        )
        # create task to write chat response to history
        asyncio.create_task(
            chat_stream_response.awrite_response_to_history(self.session.memory)
        )
        await chat_stream_response._is_function_false_event.wait()
        return chat_stream_response

    def _call_function(self, tools, function_call):
        logger.debug("in _call_function")
        function_message, tool_output = call_function(
            tools, function_call, verbose=self._verbose
        )
        self.sources.append(tool_output)
        self.session.memory.put(function_message)

    def _get_agent_response(
        self, mode: ChatMode, functions: List[dict]
    ) -> AgentChatResponse | StreamingAgentChatResponse:
        if mode == ChatMode.default:
            chat_response: ChatResponse = self._llm.chat(
                self.all_messages, functions=functions
            )
            return self._process_message(chat_response)
        elif mode == ChatMode.stream:
            return self._get_stream_ai_response(functions)

    async def _get_async_agent_response(
        self, mode: ChatMode, functions: List[dict]
    ) -> AgentChatResponse | StreamingAgentChatResponse:
        if mode == ChatMode.default:
            chat_response: ChatResponse = await self._llm.achat(
                self.all_messages, functions=functions
            )
            return self._process_message(chat_response)
        elif mode == ChatMode.stream:
            return await self._get_async_stream_ai_response(functions)

    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        *,
        mode: ChatMode = ChatMode.default,
    ) -> AgentChatResponse | StreamingAgentChatResponse:
        tools, functions = self.init_chat(message, chat_history)
        n_function_calls = 0

        # Loop until no more function calls or max_function_calls is reached
        while True:
            agent_chat_response = self._get_agent_response(mode, functions)
            latest_function = self.session.get_latest_function_call()
            if not self._should_continue(latest_function, n_function_calls):
                logger.debug("Break: should continue False")
                break
            self._call_function(tools, latest_function)
            n_function_calls += 1

        return agent_chat_response

    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        *,
        mode: ChatMode = ChatMode.default,
    ) -> AgentChatResponse | StreamingAgentChatResponse:
        tools, functions = self.init_chat(message, chat_history)
        n_function_calls = 0

        # Loop until no more function calls or max_function_calls is reached
        while True:
            agent_chat_response = await self._get_async_agent_response(mode, functions)
            latest_function = self.session.get_latest_function_call()
            if not self._should_continue(latest_function, n_function_calls):
                break
            self._call_function(tools, latest_function)
            n_function_calls += 1

        return agent_chat_response

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        return self.chat(message, chat_history, mode=ChatMode.stream)

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        return await self.achat(message, chat_history, mode=ChatMode.stream)

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


class ExperimentalOpenAIAgent(BaseOpenAIAgent):
    def __init__(
        self,
        tools: List[BaseTool],
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

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> "ExperimentalOpenAIAgent":
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
