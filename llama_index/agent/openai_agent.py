import json
from abc import abstractmethod
from queue import Queue
from threading import Thread
from typing import AsyncGenerator, Callable, Generator, List, Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.schema import BaseNode, NodeWithScore
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.llms.base import ChatMessage, ChatResponseGen
from llama_index.llms.openai import OpenAI
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.tools import BaseTool

DEFAULT_MAX_FUNCTION_CALLS = 5
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
    return ChatMessage(content=str(output), name=function_call["name"], role="function")


class StreamingChatResponse:
    """Streaming chat response to user and writing to chat history."""

    def __init__(self, chat_stream: ChatResponseGen) -> None:
        self.chat_stream = chat_stream
        self.queue: Queue = Queue()
        self.is_done = False
        self.response = ""

    def write_response_to_history(self, chat_history: List[ChatMessage]) -> None:
        final_message = ChatMessage(content="", role="assistant")
        for chat in self.chat_stream:
            final_message = chat.message
            self.queue.put_nowait(chat.delta)
        chat_history.append(final_message)
        self.is_done = True

    @property
    def response_gen(self) -> Generator[str, None, None]:
        while not self.is_done or not self.queue.empty():
            try:
                delta = self.queue.get(block=False)
                self.response += delta
                yield delta
            except Exception:
                # Queue is empty, but we're not done yet
                continue


class BaseOpenAIAgent(BaseChatEngine, BaseQueryEngine):
    """Base OpenAI Agent."""

    def __init__(
        self,
        llm: OpenAI,
        chat_history: List[ChatMessage],
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._llm = llm
        self._chat_history = chat_history
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.callback_manager = callback_manager or CallbackManager([])

    def reset(self) -> None:
        self._chat_history.clear()

    @abstractmethod
    def _get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""

    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))
        tools = self._get_tools(message)
        functions = [tool.metadata.to_openai_function() for tool in tools]

        # TODO: Support forced function call
        chat_response = self._llm.chat(chat_history, functions=functions)
        ai_message = chat_response.message
        chat_history.append(ai_message)

        n_function_calls = 0
        function_call = ai_message.additional_kwargs.get("function_call", None)
        while function_call is not None:
            if n_function_calls >= self._max_function_calls:
                print(f"Exceeded max function calls: {self._max_function_calls}.")
                break

            function_message = call_function(
                tools, function_call, verbose=self._verbose
            )
            chat_history.append(function_message)
            n_function_calls += 1

            # send function call & output back to get another response
            chat_response = self._llm.chat(chat_history, functions=functions)
            ai_message = chat_response.message
            chat_history.append(ai_message)
            function_call = ai_message.additional_kwargs.get("function_call", None)

        return Response(ai_message.content)

    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Generator[StreamingChatResponse, None, None]:
        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))
        tools = self._get_tools(message)
        functions = [tool.metadata.to_openai_function() for tool in tools]

        def _stream_response(
            message: str, chat_history: List[ChatMessage]
        ) -> Generator[StreamingChatResponse, None, None]:
            # TODO: Support forced function call
            chat_stream_response = StreamingChatResponse(
                self._llm.stream_chat(chat_history, functions=functions)
            )

            # Get the response in a separate thread so we can yield the response
            thread = Thread(
                target=chat_stream_response.write_response_to_history,
                args=(chat_history,),
            )
            thread.start()
            yield chat_stream_response
            thread.join()

            n_function_calls = 0
            function_call = chat_history[-1].additional_kwargs.get(
                "function_call", None
            )
            while function_call is not None:
                if n_function_calls >= self._max_function_calls:
                    print(f"Exceeded max function calls: {self._max_function_calls}.")
                    break

                function_message = call_function(
                    tools, function_call, verbose=self._verbose
                )
                chat_history.append(function_message)
                n_function_calls += 1

                # send function call & output back to get another response
                chat_stream_response = StreamingChatResponse(
                    self._llm.stream_chat(chat_history, functions=functions)
                )

                # Get the response in a separate thread so we can yield the response
                thread = Thread(
                    target=chat_stream_response.write_response_to_history,
                    args=(chat_history,),
                )
                thread.start()
                yield chat_stream_response
                thread.join()

                function_call = chat_history[-1].additional_kwargs.get(
                    "function_call", None
                )

        return _stream_response(message, chat_history)

    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> RESPONSE_TYPE:
        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))
        tools = self._get_tools(message)
        functions = [tool.metadata.to_openai_function() for tool in tools]

        # TODO: Support forced function call
        chat_response = await self._llm.achat(chat_history, functions=functions)
        ai_message = chat_response.message
        chat_history.append(ai_message)

        n_function_calls = 0
        function_call = ai_message.additional_kwargs.get("function_call", None)
        while function_call is not None:
            if n_function_calls >= self._max_function_calls:
                print(f"Exceeded max function calls: {self._max_function_calls}.")
                continue

            function_message = call_function(
                tools, function_call, verbose=self._verbose
            )
            chat_history.append(function_message)
            n_function_calls += 1

            # send function call & output back to get another response
            response = await self._llm.achat(chat_history, functions=functions)
            ai_message = response.message
            chat_history.append(ai_message)
            function_call = ai_message.additional_kwargs.get("function_call", None)

        return Response(ai_message.content)

    def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AsyncGenerator[StreamingChatResponse, None]:
        chat_history = chat_history or self._chat_history
        chat_history.append(ChatMessage(content=message, role="user"))
        tools = self._get_tools(message)
        functions = [tool.metadata.to_openai_function() for tool in tools]

        async def _stream_response(
            message: str, chat_history: List[ChatMessage]
        ) -> AsyncGenerator[StreamingChatResponse, None]:
            # TODO: Support forced function call
            chat_stream_response = StreamingChatResponse(
                await self._llm.astream_chat(chat_history, functions=functions)
            )

            # Get the response in a separate thread so we can yield the response
            thread = Thread(
                target=chat_stream_response.write_response_to_history,
                args=(chat_history,),
            )
            thread.start()
            yield chat_stream_response
            thread.join()

            n_function_calls = 0
            function_call = chat_history[-1].additional_kwargs.get(
                "function_call", None
            )
            while function_call is not None:
                if n_function_calls >= self._max_function_calls:
                    print(f"Exceeded max function calls: {self._max_function_calls}.")
                    break

                function_message = call_function(
                    tools, function_call, verbose=self._verbose
                )
                chat_history.append(function_message)
                n_function_calls += 1

                # send function call & output back to get another response
                chat_stream_response = StreamingChatResponse(
                    await self._llm.astream_chat(chat_history, functions=functions)
                )

                # Get the response in a separate thread so we can yield the response
                thread = Thread(
                    target=chat_stream_response.write_response_to_history,
                    args=(chat_history,),
                )
                thread.start()
                yield chat_stream_response
                thread.join()

                function_call = chat_history[-1].additional_kwargs.get(
                    "function_call", None
                )

        return _stream_response(message, chat_history)

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
        chat_history: List[ChatMessage],
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
        self._tools = tools

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[OpenAI] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "OpenAIAgent":
        tools = tools or []
        chat_history = chat_history or []
        llm = llm or OpenAI(model="gpt-3.5-turbo-0613")
        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")

        if llm.model not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Model name {llm.model} not supported. "
                f"Supported model names: {SUPPORTED_MODEL_NAMES}"
            )

        return cls(
            tools=tools,
            llm=llm,
            chat_history=chat_history,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )

    def _get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._tools


class RetrieverOpenAIAgent(BaseOpenAIAgent):
    """Retriever OpenAI Agent.

    NOTE: this is a beta feature, function interfaces might change.

    TODO: add a native OpenAI Tool Index.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        node_to_tool_fn: Callable[[BaseNode], BaseTool],
        llm: OpenAI,
        chat_history: List[ChatMessage],
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
        self._node_to_tool_fn = node_to_tool_fn

    @classmethod
    def from_retriever(
        cls,
        retriever: BaseRetriever,
        node_to_tool_fn: Callable[[BaseNode], BaseTool],
        llm: Optional[OpenAI] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "RetrieverOpenAIAgent":
        lc_chat_history = chat_history or []
        llm = llm or OpenAI(model="gpt-3.5-turbo-0613")
        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")

        if llm.model not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Model name {llm.model} not supported. "
                f"Supported model names: {SUPPORTED_MODEL_NAMES}"
            )

        return cls(
            retriever=retriever,
            node_to_tool_fn=node_to_tool_fn,
            llm=llm,
            chat_history=lc_chat_history,
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
