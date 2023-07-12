import asyncio
import json
import time
from abc import abstractmethod
from threading import Thread
from typing import (
    AsyncGenerator,
    Callable,
    Generator,
    List,
    Tuple,
    Type,
    Optional,
)

from llama_index.callbacks.base import CallbackManager
from llama_index.chat_engine.types import (
    BaseChatEngine,
    StreamingChatResponse,
    STREAMING_CHAT_RESPONSE_TYPE,
)
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.llms.base import (
    ChatMessage,
    MessageRole,
)
from llama_index.llms.openai import OpenAI
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.schema import BaseNode, NodeWithScore
from llama_index.tools import BaseTool

DEFAULT_MAX_FUNCTION_CALLS = 5
DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"
SUPPORTED_MODEL_NAMES = [
    "gpt-3.5-turbo-0613",
    "gpt-4-0613",
]


def get_function_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    """Get function by name."""
    name_to_tool = {tool.metadata.name: tool for tool in tools}
    if name not in name_to_tool:
        raise ValueError(f"Tool with name {name} not found")
    return name_to_tool[name]


def call_function(
    tools: List[BaseTool], function_call: dict, verbose: bool = False
) -> ChatMessage:
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
        print(f"Got output: {output}")
        print("========================")
    return ChatMessage(
        content=str(output),
        role=MessageRole.FUNCTION,
        additional_kwargs={
            "name": function_call["name"],
        },
    )


class BaseOpenAIAgent(BaseChatEngine, BaseQueryEngine):
    """Base OpenAI Agent."""

    def __init__(
        self,
        llm: OpenAI,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.callback_manager = callback_manager or CallbackManager([])

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self._memory.get_all()

    def reset(self) -> None:
        self._memory.reset()

    @abstractmethod
    def _get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""

    def _get_latest_function_call(
        self, chat_history: List[ChatMessage]
    ) -> Optional[dict]:
        """Get latest function call from chat history."""
        return chat_history[-1].additional_kwargs.get("function_call", None)

    def _init_chat(self, message: str) -> Tuple[List[BaseTool], List[dict]]:
        """Add user message to chat history and get tools and functions."""
        self._memory.put(ChatMessage(content=message, role=MessageRole.USER))
        tools = self._get_tools(message)
        functions = [tool.metadata.to_openai_function() for tool in tools]
        return tools, functions

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        if chat_history is not None:
            self._memory.set(chat_history)

        tools, functions = self._init_chat(message)

        # TODO: Support forced function call
        all_messages = self._prefix_messages + self._memory.get()
        chat_response = self._llm.chat(all_messages, functions=functions)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        n_function_calls = 0
        function_call = self._get_latest_function_call(self._memory.get_all())
        while function_call is not None:
            if n_function_calls >= self._max_function_calls:
                print(f"Exceeded max function calls: {self._max_function_calls}.")
                break

            function_message = call_function(
                tools, function_call, verbose=self._verbose
            )
            self._memory.put(function_message)
            n_function_calls += 1

            # send function call & output back to get another response
            all_messages = self._prefix_messages + self._memory.get()
            chat_response = self._llm.chat(all_messages, functions=functions)
            ai_message = chat_response.message
            self._memory.put(ai_message)
            function_call = self._get_latest_function_call(self._memory.get_all())

        return Response(ai_message.content)

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> STREAMING_CHAT_RESPONSE_TYPE:
        if chat_history is not None:
            self._memory.set(chat_history)
        tools, functions = self._init_chat(message)

        def gen() -> Generator[StreamingChatResponse, None, None]:
            # TODO: Support forced function call
            all_messages = self._prefix_messages + self._memory.get()
            chat_stream_response = StreamingChatResponse(
                self._llm.stream_chat(all_messages, functions=functions)
            )

            # Get the response in a separate thread so we can yield the response
            thread = Thread(
                target=chat_stream_response.write_response_to_history,
                args=(chat_history,),
            )
            thread.start()
            yield chat_stream_response

            while chat_stream_response._is_function is None:
                # Wait until we know if the response is a function call or not
                time.sleep(0.05)
                if chat_stream_response._is_function is False:
                    return

            thread.join()

            n_function_calls = 0
            function_call = self._get_latest_function_call(self._memory.get_all())
            while function_call is not None:
                if n_function_calls >= self._max_function_calls:
                    print(f"Exceeded max function calls: {self._max_function_calls}.")
                    break

                function_message = call_function(
                    tools, function_call, verbose=self._verbose
                )
                self._memory.put(function_message)
                n_function_calls += 1

                all_messages = self._prefix_messages + self._memory.get()
                # send function call & output back to get another response
                chat_stream_response = StreamingChatResponse(
                    self._llm.stream_chat(all_messages, functions=functions)
                )

                # Get the response in a separate thread so we can yield the response
                thread = Thread(
                    target=chat_stream_response.write_response_to_history,
                    args=(self._memory,),
                )
                thread.start()
                yield chat_stream_response

                while chat_stream_response._is_function is None:
                    # Wait until we know if the response is a function call or not
                    time.sleep(0.05)
                    if chat_stream_response._is_function is False:
                        return

                thread.join()
                function_call = self._get_latest_function_call(self._memory.get_all())

        return gen()

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        if chat_history is not None:
            self._memory.set(chat_history)

        tools, functions = self._init_chat(message)

        # TODO: Support forced function call
        all_messages = self._prefix_messages + self._memory.get()
        chat_response = await self._llm.achat(all_messages, functions=functions)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        n_function_calls = 0
        function_call = self._get_latest_function_call(self._memory.get_all())
        while function_call is not None:
            if n_function_calls >= self._max_function_calls:
                print(f"Exceeded max function calls: {self._max_function_calls}.")
                continue

            function_message = call_function(
                tools, function_call, verbose=self._verbose
            )
            self._memory.put(function_message)
            n_function_calls += 1

            # send function call & output back to get another response
            response = await self._llm.achat(
                self._prefix_messages + self._memory.get(), functions=functions
            )
            ai_message = response.message
            self._memory.put(ai_message)
            function_call = self._get_latest_function_call(self._memory.get_all())

        return Response(ai_message.content)

    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> STREAMING_CHAT_RESPONSE_TYPE:
        if chat_history is not None:
            self._memory.set(chat_history)

        tools, functions = self._init_chat(message)

        async def gen() -> AsyncGenerator[StreamingChatResponse, None]:
            all_messages = self._prefix_messages + self._memory.get()
            # TODO: Support forced function call
            chat_stream_response = StreamingChatResponse(
                await self._llm.astream_chat(all_messages, functions=functions)
            )

            # Get the response in a separate thread so we can yield the response
            thread = Thread(
                target=lambda x: asyncio.run(
                    chat_stream_response.awrite_response_to_history(x)
                ),
                args=(self._memory,),
            )
            thread.start()
            yield chat_stream_response

            while chat_stream_response._is_function is None:
                # Wait until we know if the response is a function call or not
                time.sleep(0.05)
                if chat_stream_response._is_function is False:
                    return

            thread.join()

            n_function_calls = 0
            function_call = self._get_latest_function_call(self._memory.get_all())
            while function_call is not None:
                if n_function_calls >= self._max_function_calls:
                    print(f"Exceeded max function calls: {self._max_function_calls}.")
                    break

                function_message = call_function(
                    tools, function_call, verbose=self._verbose
                )
                self._memory.put(function_message)
                n_function_calls += 1

                # send function call & output back to get another response
                all_messages = self._prefix_messages + self._memory.get()
                chat_stream_response = StreamingChatResponse(
                    await self._llm.astream_chat(all_messages, functions=functions)
                )

                # Get the response in a separate thread so we can yield the response
                thread = Thread(
                    target=lambda x: asyncio.run(
                        chat_stream_response.awrite_response_to_history(x)
                    ),
                    args=(self._memory,),
                )
                thread.start()
                yield chat_stream_response

                while chat_stream_response._is_function is None:
                    # Wait until we know if the response is a function call or not
                    time.sleep(0.05)
                    if chat_stream_response._is_function is False:
                        return

                thread.join()
                function_call = self._get_latest_function_call(self._memory.get_all())

        return gen()

    # ===== Query Engine Interface =====
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return self.chat(
            query_bundle.query_str,
            chat_history=[],
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return await self.achat(
            query_bundle.query_str,
            chat_history=[],
        )


class OpenAIAgent(BaseOpenAIAgent):
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
        llm: Optional[OpenAI] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
    ) -> "OpenAIAgent":
        tools = tools or []
        chat_history = chat_history or []
        memory = memory or memory_cls.from_defaults(chat_history)
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
