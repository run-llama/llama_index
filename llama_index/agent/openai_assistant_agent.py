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
from llama_index.llms.openai_utils import from_openai_messages
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.objects.base import ObjectRetriever
from llama_index.tools import BaseTool, ToolOutput, adapt_to_async_tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def from_openai_thread_message(thread_message: Any) -> ChatMessage:
    """From OpenAI thread message."""
    from openai.types.beta.threads import ThreadMessage, MessageContentText
    thread_message = cast(ThreadMessage, thread_message)

    # we don't have a way of showing images, just do text for now
    text_contents = [t for t in thread_message.content if isinstance(t, MessageContentText)]
    text_content_str = " ".join([t.text.value for t in text_contents])

    return ChatMessage(
        role=thread_message.role,
        content=text_content_str,
        additional_kwargs={
            "thread_message": thread_message,
            "thread_id": thread_message.thread_id,
            "assistant_id": thread_message.assistant_id,
            "id": thread_message.id,
            "metadata": thread_message.metadata,
        }
    )


def from_openai_thread_messages(thread_messages: List[Any]) -> List[ChatMessage]:
    """From OpenAI thread messages."""
    return [from_openai_thread_message(thread_message) for thread_message in thread_messages]
    



class OpenAIAssistantAgent(BaseAgent):
    """OpenAIAssistant agent.

    Wrapper around OpenAI assistant API: https://platform.openai.com/docs/assistants/overview
    
    """

    def __init__(
        self, 
        client: Any,
        assistant: Any, 
        tools: List[BaseTool],
        callback_manager: Optional[CallbackManager] = None,
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

        self.callback_manager = callback_manager or CallbackManager([])

    @classmethod
    def from_new(
        cls,
        name: str,
        instructions: str,
        tools: Optional[List[BaseTool]] = None,
        openai_tools: Optional[List[Dict]] = None,
        thread_id: Optional[str] = None,
        model: str = "gpt-4-1106-preview",
        instructions_prefix: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "OpenAIAssistantAgent":
        """From new."""
        from openai import OpenAI

        # this is the set of openai tools
        # not to be confused with the tools we pass in for function calling
        openai_tools = openai_tools or [{"type": "code_interpreter"}]
        client = OpenAI()
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=openai_tools,
            model=model
        )
        return cls(
            client, assistant, tools, 
            callback_manager=callback_manager,
            thread_id=thread_id, instructions_prefix=instructions_prefix
        )

    @property
    def chat_history(self) -> List[ChatMessage]:
        raw_messages = self._client.beta.threads.messages.list(
            thread_id=self._thread_id
        )
        messages = from_openai_thread_messages(raw_messages)
        return messages

    def reset(self) -> None:
        """Delete and create a new thread."""
        self._client.beta.threads.delete(self._thread_id)
        thread = self._client.beta.threads.create()
        thread_id = thread.id
        self._thread_id = thread_id

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
        while run.status in ["queued", "in_progress"]:
            run = self._client.beta.threads.runs.retrieve(
                thread_id=self._thread_id,
                run_id=run.id
            )
            time.sleep(self._run_retrieve_sleep_time)
        if run.status != "completed":
            raise ValueError(
                f"Run failed with status {run.status}.\n"
                f"Error: {run.last_error}"
            )
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
        messages = from_openai_thread_messages(raw_messages)
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

