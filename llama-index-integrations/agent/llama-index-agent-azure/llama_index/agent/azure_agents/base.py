from typing import List, Optional, Union
import os
import time
from llama_index.core.agent.types import BaseAgent
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.core.callbacks import CallbackManager, trace_method, CBEventType, EventPayload
 
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

class AzureAgent(BaseAgent):
    def __init__(
        self,
        project_connection_string: str,
        api_version: str,
        model: str = "gpt-4o-mini",
        name: str = "azure-agent",
        instructions: str = "You are a helpful agent",
        thread_id: Optional[str] = None,
        verbose: bool = False,
        run_retrieve_sleep_time: float = 1.0,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self._project_connection_string = project_connection_string
        self._api_version = api_version
        self._model = model
        self._name = name
        self._instructions = instructions
        self._verbose = verbose
        self._session_id = None
        self._run_retrieve_sleep_time = run_retrieve_sleep_time
        self.callback_manager = callback_manager or CallbackManager([])
        # Instantiate Azure client once
        self._project_client = AIProjectClient.from_connection_string(
            conn_str=self._project_connection_string,
            credential=DefaultAzureCredential()
        )
        self._client: AzureOpenAI = self._project_client.inference.get_azure_openai_client(
            api_version=self._api_version
        )
        # Create or use thread
        if thread_id is not None:
            self._thread_id = thread_id
        else:
            with self._client:
                thread = self._client.beta.threads.create()
                self._thread_id = thread.id
        self._assistant = None  # Will be created on first chat

    def run_assistant(self) -> None:
        """
        Run the assistant and poll until completion. Returns the run object.
        (No tool support for now, so just polls until done or failed.)
        """
        run = self._client.beta.threads.runs.create(
            thread_id=self._thread_id,
            assistant_id=self._assistant.id,
        )
        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(self._run_retrieve_sleep_time)
            run = self._client.beta.threads.runs.retrieve(
                thread_id=self._thread_id, run_id=run.id
            )
            if self._verbose:
                print(f"Run status: {run.status}")
        if run.status == "failed":
            raise ValueError(f"Run failed with status {run.status}.")
        return run

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs
    ) -> AgentChatResponse:
        """
        Internal chat logic for Azure OpenAI Assistants API.
        """
        with self._client:
            # Create assistant if not already created
            if self._assistant is None:
                self._assistant = self._client.beta.assistants.create(
                    model=self._model, name=self._name, instructions=self._instructions
                )
                if self._verbose:
                    print(f"Created agent, agent ID: {self._assistant.id}")
            thread_id = self._thread_id
            msg = self._client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=message
            )
            if self._verbose:
                print(f"Created message, message ID: {msg.id}")
            run = self.run_assistant()
            messages = self._client.beta.threads.messages.list(thread_id=thread_id)
            last_message = messages.data[-1] if messages.data else None
            response_text = ""
            if last_message and hasattr(last_message, "content") and last_message.content:
                for block in last_message.content:
                    if block.__class__.__name__ == "TextContentBlock":
                        text_obj = getattr(block, "text", None)
                        if text_obj is not None:
                            text_val = getattr(text_obj, "value", None)
                            if text_val:
                                response_text = text_val
                                break
            return AgentChatResponse(response=response_text)

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
            trace=True,
        ) as e:
            chat_response = self._chat(message, chat_history, **kwargs)
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None, **kwargs
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("stream_chat not implemented")

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        raise NotImplementedError("achat not implemented")

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("astream_chat not implemented")

    @staticmethod
    def from_openai_thread_message(thread_message: object) -> ChatMessage:
        """Convert an OpenAI/Azure thread message to ChatMessage."""
        # Import types dynamically to avoid hard dependency
        try:
            from openai.types.beta.threads import TextContentBlock, Message
        except ImportError:
            raise ImportError("openai.types.beta.threads is required for this method.")
        thread_message = thread_message  # No cast needed in Python, just use as is
        text_contents = [
            t for t in getattr(thread_message, 'content', []) if t.__class__.__name__ == "TextContentBlock"
        ]
        text_content_str = " ".join([getattr(getattr(t, 'text', None), 'value', '') for t in text_contents])
        return ChatMessage(
            role=getattr(thread_message, 'role', ''),
            content=text_content_str,
            additional_kwargs={
                "thread_message": thread_message,
                "thread_id": getattr(thread_message, 'thread_id', None),
                "assistant_id": getattr(thread_message, 'assistant_id', None),
                "id": getattr(thread_message, 'id', None),
                "metadata": getattr(thread_message, 'metadata', None),
            },
        )

    @staticmethod
    def from_openai_thread_messages(thread_messages: list) -> List[ChatMessage]:
        """Convert a list of OpenAI/Azure thread messages to ChatMessage list."""
        return [AzureAgent.from_openai_thread_message(m) for m in thread_messages]

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Return the chat history as a list of ChatMessage objects."""
        messages = self._client.beta.threads.messages.list(thread_id=self._thread_id)
        return self.from_openai_thread_messages(messages.data)

    @property
    def thread_id(self) -> str:
        return self._thread_id

    @property
    def client(self) -> AzureOpenAI:
        return self._client

    @property
    def assistant(self):
        return self._assistant

    @property
    def last_message(self) -> Optional[ChatMessage]:
        messages = self._client.beta.threads.messages.list(thread_id=self._thread_id)
        if messages.data:
            return self.from_openai_thread_message(messages.data[-1])
        return None

    @property
    def get_session_id(self) -> str:
        return self._session_id or ""
