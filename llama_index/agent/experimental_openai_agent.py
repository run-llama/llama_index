import asyncio
import json
import time
from abc import abstractmethod
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple, Type

from llama_index.agent.types import BaseAgent
from llama_index.callbacks.base import CallbackManager
from llama_index.chat_engine.types import (AgentChatResponse,
                                           StreamingAgentChatResponse)
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
    def __init__(self, llm, memory, prefix_messages, verbose, max_function_calls, callback_manager, response_handler, stream_handler):
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._verbose = verbose
        self._max_function_calls = max_function_calls
        self.callback_manager = callback_manager or CallbackManager([])
        self.response_handler = response_handler
        self.stream_handler = stream_handler

    def _get_latest_function_call(self) -> Optional[dict]:
        """Get latest function call from chat history."""
        return self._memory[-1].additional_kwargs.get("function_call", None)

    def init_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None):
        session = ChatSession(self._memory, self._prefix_messages)
        tools, functions = session.init_chat(message, chat_history)
        latest_function_call = self._get_latest_function_call()
        return session, tools, functions, latest_function_call

    def handle_ai_response(self, session, tools, functions):
        all_messages = session.get_all_messages()
        chat_response = self.response_handler.get_response(all_messages, functions)
        ai_message = chat_response.message
        session.memory.put(ai_message)

        function_call = session.get_latest_function_call()
        function_message, sources = None, []
        if function_call is not None:
            function_message, sources = call_function(tools, function_call, verbose=self._verbose)
            session.memory.put(function_message)

        return ai_message, function_call, sources

    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> AgentChatResponse:
        session, tools, functions, function_call = self.init_chat(message, chat_history)

        n_function_calls = 0
        while function_call is not None and n_function_calls < self._max_function_calls:
            ai_message, function_call, sources = self.handle_ai_response(session, tools, functions)
            n_function_calls += 1

        return AgentChatResponse(response=str(ai_message.content), sources=sources)

    def stream_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> StreamingAgentChatResponse:
        session, tools, functions, function_call = self.init_chat(message, chat_history)

        n_function_calls = 0
        while function_call is not None and n_function_calls < self._max_function_calls:
            ai_message, function_call, sources = self.handle_ai_response(session, tools, functions)
            n_function_calls += 1

        return StreamingAgentChatResponse(response=str(ai_message.content), sources=sources)

    async def achat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> AgentChatResponse:
        session, tools, functions, function_call = self.init_chat(message, chat_history)

        n_function_calls = 0
        while function_call is not None and n_function_calls < self._max_function_calls:
            ai_message, function_call, sources = await self.handle_ai_response(session, tools, functions)
            n_function_calls += 1

        return AgentChatResponse(response=str(ai_message.content), sources=sources)

    async def astream_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> StreamingAgentChatResponse:
        session, tools, functions, function_call = self.init_chat(message, chat_history)

        n_function_calls = 0
        while function_call is not None and n_function_calls < self._max_function_calls:
            ai_message, function_call, sources = await self.handle_ai_response(session, tools, functions)
            n_function_calls += 1

        return StreamingAgentChatResponse(response=str(ai_message.content), sources=sources)


from dataclasses import dataclass
from queue import Queue
from typing import List, Optional


class MessageRole:
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

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

class BaseTool:
    pass

class ToolOutput:
    pass

class BaseAgent:
    pass

class OpenAI:
    pass

class CallbackManager:
    pass

class ChatResponseGen:
    pass

class ChatResponseAsyncGen:
    pass

class ChatSession:
    def __init__(self, memory: BaseMemory, prefix_messages: List[ChatMessage]):
        self.memory = memory
        self.prefix_messages = prefix_messages
        self.tools = []
        self.functions = []

    def init_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> Tuple[List[BaseTool], List[dict]]:
        if chat_history is not None:
            self.memory.set(chat_history)
        self.memory.put(ChatMessage(content=message, role=MessageRole.USER))
        self.tools = self._get_tools(message)
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
