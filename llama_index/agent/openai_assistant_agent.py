"""OpenAI Assistant Agent."""
import asyncio
import json
import logging
from abc import abstractmethod
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast
import time

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
from llama_index.llms.base import ChatMessage
# from llama_index.llms.openai_utils import is_function_calling_model, from_openai_message_dicts
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.objects.base import ObjectRetriever
from llama_index.tools import BaseTool, ToolOutput, adapt_to_async_tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)



import openai


class OpenAIAssistantAgent(BaseAgent):
    """OpenAIAssistant agent.

    Wrapper around OpenAI assistant API: https://platform.openai.com/docs/assistants/overview
    
    """

    def __init__(
        self, 
        client: Any,
        assistant: Any, 
        tools: List[BaseTool],
        thread_id: Optional[str] = None,
        instructions_prefix: Optional[str] = None,
        run_retrieve_sleep_time: float = 0.1,
    ) -> None:
        """Init params."""
        from openai import OpenAI
        from openai.types.beta.assistant import Assistant

        self._client = cast(OpenAI, client)
        self._assistant = cast(Assistant, assistant)
        self._tools = tools
        if thread_id is None:
            thread = self._client.beta.threads.create()
            thread_id = thread.id
        self._thread_id = thread_id
        self._instructions_prefix = instructions_prefix
        self._run_retrieve_sleep_time = run_retrieve_sleep_time

    @classmethod
    def from_new(
        cls,
        name: str,
        instructions: str,
        tools: Optional[List[BaseTool]] = None,
        openai_tool_names: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        model: str = "gpt-4-1106-preview",
        instructions_prefix: Optional[str] = None,
    ) -> "OpenAIAssistantAgent":
        """From new."""
        from openai import OpenAI

        # this is the set of openai tools
        # not to be confused with the tools we pass in for function calling
        openai_tool_names = openai_tool_names or ["code_interpreter"]
        client = OpenAI()
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=openai_tool_names,
            model=model
        )
        return cls(
            client, assistant, tools, thread_id=thread_id, instructions_prefix=instructions_prefix
        )
        

    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._tools

    def add_message(self, message: str) -> Any:
        """Add message to assistant."""
        message = self._client.beta.threads.messages.create(
            thread_id=self._thread_id,
            role="user",
            content=message
        )
        return message

    def run_assistant(self, instructions_prefix: Optional[str] = None) -> Any:
        """Run assistant."""
        instructions_prefix = instructions_prefix or self._instructions_prefix
        run = self._client.beta.threads.runs.create(
            thread_id=self._thread_id,
            assistant_id=self._assistant.id,
            instructions=self._instructions_prefix
        )
        run = self.retrieve_response(run)
        return run

    def retrieve_response(self, run: Any) -> Any:
        """Retrieve response from assistant."""
        from openai.types.beta.threads import Run
        run = cast(Run, run)
        while run.status == "queued":
            run = self._client.beta.threads.runs.retrieve(
                thread_id=self._thread_id,
                run_id=run.id
            )
            time.sleep(self._run_retrieve_sleep_time)
        return run

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Main chat interface."""
        added_message_obj = self.add_message(message)
        run = self.run_assistant(
            instructions_prefix=self._instructions_prefix
        )
        raw_messages = self._client.beta.threads.messages.list(
            thread_id=self._thread_id
        )
        messages = from_openai_message_dicts(raw_messages)
        return messages

    async def _achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        raise NotImplementedError("async chat not implemented")

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
        raise NotImplementedError("stream_chat not implemented")

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("astream_chat not implemented")

