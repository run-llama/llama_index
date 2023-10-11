import asyncio
import json
import logging
from abc import abstractmethod
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

from llama_index.agent.types import BaseAgent
from llama_index.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.llms.base import LLM, ChatMessage, ChatResponse, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_utils import is_function_calling_model
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.objects.base import ObjectRetriever
from llama_index.tools import BaseTool, ToolOutput, adapt_to_async_tool

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
        print(f"Got output: {output!s}")
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


async def acall_function(
    tools: List[BaseTool], function_call: dict, verbose: bool = False
) -> Tuple[ChatMessage, ToolOutput]:
    """Call a function and return the output as a string."""
    name = function_call["name"]
    arguments_str = function_call["arguments"]
    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = get_function_by_name(tools, name)
    async_tool = adapt_to_async_tool(tool)
    argument_dict = json.loads(arguments_str)
    output = await async_tool.acall(**argument_dict)
    if verbose:
        print(f"Got output: {output!s}")
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


def resolve_function_call(function_call: Union[str, dict] = "auto") -> Union[str, dict]:
    """Resolve function call.

    If function_call is a function name string, return a dict with the name.
    """
    if isinstance(function_call, str) and function_call not in ["none", "auto"]:
        return {"name": function_call}

    return function_call


class BaseOpenAIAgent(BaseAgent):
    def __init__(
        self,
        llm: OpenAI,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        verbose: bool,
        max_function_calls: int,
        callback_manager: Optional[CallbackManager],
    ):
        self._llm = llm
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.prefix_messages = prefix_messages
        self.memory = memory
        self.callback_manager = callback_manager or self._llm.callback_manager
        self.sources: List[ToolOutput] = []

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self.memory.get_all()

    @property
    def all_messages(self) -> List[ChatMessage]:
        return self.prefix_messages + self.memory.get()

    @property
    def latest_function_call(self) -> Optional[dict]:
        return self.memory.get_all()[-1].additional_kwargs.get("function_call", None)

    def reset(self) -> None:
        self.memory.reset()

    @abstractmethod
    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""

    def _should_continue(
        self, function_call: Optional[dict], n_function_calls: int
    ) -> bool:
        if n_function_calls > self._max_function_calls:
            return False
        if not function_call:
            return False
        return True

    def init_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Tuple[List[BaseTool], List[dict]]:
        if chat_history is not None:
            self.memory.set(chat_history)
        self.sources = []
        self.memory.put(ChatMessage(content=message, role=MessageRole.USER))
        tools = self.get_tools(message)
        functions = [tool.metadata.to_openai_function() for tool in tools]
        return tools, functions

    def _process_message(self, chat_response: ChatResponse) -> AgentChatResponse:
        ai_message = chat_response.message
        self.memory.put(ai_message)
        return AgentChatResponse(response=str(ai_message.content), sources=self.sources)

    def _get_stream_ai_response(
        self, **llm_chat_kwargs: Any
    ) -> StreamingAgentChatResponse:
        chat_stream_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(**llm_chat_kwargs),
            sources=self.sources,
        )
        # Get the response in a separate thread so we can yield the response
        thread = Thread(
            target=chat_stream_response.write_response_to_history,
            args=(self.memory,),
        )
        thread.start()
        # Wait for the event to be set
        chat_stream_response._is_function_not_none_thread_event.wait()
        # If it is executing an openAI function, wait for the thread to finish
        if chat_stream_response._is_function:
            thread.join()
        # if it's false, return the answer (to stream)
        return chat_stream_response

    async def _get_async_stream_ai_response(
        self, **llm_chat_kwargs: Any
    ) -> StreamingAgentChatResponse:
        chat_stream_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(**llm_chat_kwargs),
            sources=self.sources,
        )
        # create task to write chat response to history
        asyncio.create_task(
            chat_stream_response.awrite_response_to_history(self.memory)
        )
        # wait until openAI functions stop executing
        await chat_stream_response._is_function_false_event.wait()
        # return response stream
        return chat_stream_response

    def _call_function(self, tools: List[BaseTool], function_call: dict) -> None:
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: function_call["arguments"],
                EventPayload.TOOL: get_function_by_name(
                    tools, function_call["name"]
                ).metadata,
            },
        ) as event:
            function_message, tool_output = call_function(
                tools, function_call, verbose=self._verbose
            )
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        self.sources.append(tool_output)
        self.memory.put(function_message)

    async def _acall_function(self, tools: List[BaseTool], function_call: dict) -> None:
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: function_call["arguments"],
                EventPayload.TOOL: get_function_by_name(
                    tools, function_call["name"]
                ).metadata,
            },
        ) as event:
            function_message, tool_output = await acall_function(
                tools, function_call, verbose=self._verbose
            )
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})
        self.sources.append(tool_output)
        self.memory.put(function_message)

    def _get_llm_chat_kwargs(
        self, functions: List[dict], function_call: Union[str, dict] = "auto"
    ) -> Dict[str, Any]:
        llm_chat_kwargs: dict = {"messages": self.all_messages}
        if functions:
            llm_chat_kwargs.update(
                functions=functions, function_call=resolve_function_call(function_call)
            )
        return llm_chat_kwargs

    def _get_agent_response(
        self, mode: ChatResponseMode, **llm_chat_kwargs: Any
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        if mode == ChatResponseMode.WAIT:
            chat_response: ChatResponse = self._llm.chat(**llm_chat_kwargs)
            return self._process_message(chat_response)
        elif mode == ChatResponseMode.STREAM:
            return self._get_stream_ai_response(**llm_chat_kwargs)
        else:
            raise NotImplementedError

    async def _get_async_agent_response(
        self, mode: ChatResponseMode, **llm_chat_kwargs: Any
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        if mode == ChatResponseMode.WAIT:
            chat_response: ChatResponse = await self._llm.achat(**llm_chat_kwargs)
            return self._process_message(chat_response)
        elif mode == ChatResponseMode.STREAM:
            return await self._get_async_stream_ai_response(**llm_chat_kwargs)
        else:
            raise NotImplementedError

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        tools, functions = self.init_chat(message, chat_history)
        n_function_calls = 0

        # Loop until no more function calls or max_function_calls is reached
        current_func = function_call
        while True:
            llm_chat_kwargs = self._get_llm_chat_kwargs(functions, current_func)
            agent_chat_response = self._get_agent_response(mode=mode, **llm_chat_kwargs)
            if not self._should_continue(self.latest_function_call, n_function_calls):
                logger.debug("Break: should continue False")
                break
            assert isinstance(self.latest_function_call, dict)
            self._call_function(tools, self.latest_function_call)
            # change function call to the default value, if a custom function was given
            # as an argument (none and auto are predefined by OpenAI)
            if current_func not in ("auto", "none"):
                current_func = "auto"
            n_function_calls += 1

        return agent_chat_response

    async def _achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        tools, functions = self.init_chat(message, chat_history)
        n_function_calls = 0

        # Loop until no more function calls or max_function_calls is reached
        current_func = function_call
        while True:
            llm_chat_kwargs = self._get_llm_chat_kwargs(functions, current_func)
            agent_chat_response = await self._get_async_agent_response(
                mode=mode, **llm_chat_kwargs
            )
            if not self._should_continue(self.latest_function_call, n_function_calls):
                break
            assert isinstance(self.latest_function_call, dict)
            await self._acall_function(tools, self.latest_function_call)
            # change function call to the default value, if a custom function was given
            # as an argument (none and auto are predefined by OpenAI)
            if current_func not in ("auto", "none"):
                current_func = "auto"
            n_function_calls += 1

        return agent_chat_response

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, function_call, mode=ChatResponseMode.WAIT
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = await self._achat(
                message, chat_history, function_call, mode=ChatResponseMode.WAIT
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, function_call, mode=ChatResponseMode.STREAM
            )
            assert isinstance(chat_response, StreamingAgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = await self._achat(
                message, chat_history, function_call, mode=ChatResponseMode.STREAM
            )
            assert isinstance(chat_response, StreamingAgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response


class OpenAIAgent(BaseOpenAIAgent):
    """OpenAI (function calling) Agent.

    Uses the OpenAI function API to reason about whether to
    use a tool, and returning the response to the user.

    Supports both a flat list of tools as well as retrieval over the tools.

    Args:
        tools (List[BaseTool]): List of tools to use.
        llm (OpenAI): OpenAI instance.
        memory (BaseMemory): Memory to use.
        prefix_messages (List[ChatMessage]): Prefix messages to use.
        verbose (Optional[bool]): Whether to print verbose output. Defaults to False.
        max_function_calls (Optional[int]): Maximum number of function calls.
            Defaults to DEFAULT_MAX_FUNCTION_CALLS.
        callback_manager (Optional[CallbackManager]): Callback manager to use.
            Defaults to None.
        tool_retriever (ObjectRetriever[BaseTool]): Object retriever to retrieve tools.


    """

    def __init__(
        self,
        tools: List[BaseTool],
        llm: OpenAI,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )
        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            # no tools
            self._get_tools = lambda _: []

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
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
    ) -> "OpenAIAgent":
        """Create an OpenAIAgent from a list of tools.

        Similar to `from_defaults` in other classes, this method will
        infer defaults for a variety of parameters, including the LLM,
        if they are not specified.

        """
        tools = tools or []

        chat_history = chat_history or []
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")

        if callback_manager is not None:
            llm.callback_manager = callback_manager

        memory = memory or memory_cls.from_defaults(chat_history, llm=llm)

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
            tool_retriever=tool_retriever,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
        )

    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._get_tools(message)
