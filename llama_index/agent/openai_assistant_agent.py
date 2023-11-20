"""OpenAI Assistant Agent."""
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from llama_index.agent.openai_agent import get_function_by_name
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
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.tools import BaseTool, ToolOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def from_openai_thread_message(thread_message: Any) -> ChatMessage:
    """From OpenAI thread message."""
    from openai.types.beta.threads import MessageContentText, ThreadMessage

    thread_message = cast(ThreadMessage, thread_message)

    # we don't have a way of showing images, just do text for now
    text_contents = [
        t for t in thread_message.content if isinstance(t, MessageContentText)
    ]
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
        },
    )


def from_openai_thread_messages(thread_messages: List[Any]) -> List[ChatMessage]:
    """From OpenAI thread messages."""
    return [
        from_openai_thread_message(thread_message) for thread_message in thread_messages
    ]


def call_function(
    tools: List[BaseTool], fn_obj: Any, verbose: bool = False
) -> Tuple[ChatMessage, ToolOutput]:
    """Call a function and return the output as a string."""
    from openai.types.beta.threads.required_action_function_tool_call import Function

    fn_obj = cast(Function, fn_obj)
    # TMP: consolidate with other abstractions
    name = fn_obj.name
    arguments_str = fn_obj.arguments
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
                "name": fn_obj.name,
            },
        ),
        output,
    )


def _process_files(client: Any, files: List[str]) -> Dict[str, Any]:
    """Process files."""
    from openai import OpenAI

    client = cast(OpenAI, client)

    file_dict = {}
    for file in files:
        file_obj = client.files.create(file=open(file, "rb"), purpose="assistants")
        file_dict[file_obj.id] = file
    return file_dict


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
        verbose: bool = False,
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
        self._verbose = verbose

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
        run_retrieve_sleep_time: float = 0.1,
        files: Optional[List[str]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> "OpenAIAssistantAgent":
        """From new assistant.

        Args:
            name: name of assistant
            instructions: instructions for assistant
            tools: list of tools
            openai_tools: list of openai tools
            thread_id: thread id
            model: model
            run_retrieve_sleep_time: run retrieve sleep time
            instructions_prefix: instructions prefix
            callback_manager: callback manager
            verbose: verbose

        """
        from openai import OpenAI

        # this is the set of openai tools
        # not to be confused with the tools we pass in for function calling
        openai_tools = openai_tools or []
        tools = tools or []
        tool_fns = [t.metadata.to_openai_tool() for t in tools]
        all_openai_tools = openai_tools + tool_fns

        # initialize client
        client = OpenAI()

        # process files
        files = files or []
        file_dict = _process_files(client, files)

        # TODO: openai's typing is a bit sus
        all_openai_tools = cast(List[Any], all_openai_tools)
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=cast(List[Any], all_openai_tools),
            model=model,
            file_ids=list(file_dict.keys()),
        )
        return cls(
            client,
            assistant,
            tools,
            callback_manager=callback_manager,
            thread_id=thread_id,
            instructions_prefix=instructions_prefix,
            verbose=verbose,
        )

    @property
    def assistant(self) -> Any:
        """Get assistant."""
        return self._assistant

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    @property
    def thread_id(self) -> str:
        """Get thread id."""
        return self._thread_id

    @property
    def chat_history(self) -> List[ChatMessage]:
        raw_messages = self._client.beta.threads.messages.list(
            thread_id=self._thread_id, order="asc"
        )
        return from_openai_thread_messages(list(raw_messages))

    def reset(self) -> None:
        """Delete and create a new thread."""
        self._client.beta.threads.delete(self._thread_id)
        thread = self._client.beta.threads.create()
        thread_id = thread.id
        self._thread_id = thread_id

    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._tools

    def upload_files(self, files: List[str]) -> Dict[str, Any]:
        """Upload files."""
        return _process_files(self._client, files)

    def add_message(self, message: str, file_ids: Optional[List[str]] = None) -> Any:
        """Add message to assistant."""
        file_ids = file_ids or []
        return self._client.beta.threads.messages.create(
            thread_id=self._thread_id,
            role="user",
            content=message,
            file_ids=file_ids,
        )

    def _run_function_calling(self, run: Any) -> List[ToolOutput]:
        """Run function calling."""
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tool_output_dicts = []
        tool_output_objs: List[ToolOutput] = []
        for tool_call in tool_calls:
            fn_obj = tool_call.function
            _, tool_output = call_function(self._tools, fn_obj, verbose=self._verbose)
            tool_output_dicts.append(
                {"tool_call_id": tool_call.id, "output": str(tool_output)}
            )
            tool_output_objs.append(tool_output)

        # submit tool outputs
        # TODO: openai's typing is a bit sus
        self._client.beta.threads.runs.submit_tool_outputs(
            thread_id=self._thread_id,
            run_id=run.id,
            tool_outputs=cast(List[Any], tool_output_dicts),
        )
        return tool_output_objs

    def run_assistant(
        self, instructions_prefix: Optional[str] = None
    ) -> Tuple[Any, Dict]:
        """Run assistant."""
        instructions_prefix = instructions_prefix or self._instructions_prefix
        run = self._client.beta.threads.runs.create(
            thread_id=self._thread_id,
            assistant_id=self._assistant.id,
            instructions=self._instructions_prefix,
        )
        from openai.types.beta.threads import Run

        run = cast(Run, run)

        sources = []

        while run.status in ["queued", "in_progress", "requires_action"]:
            run = self._client.beta.threads.runs.retrieve(
                thread_id=self._thread_id, run_id=run.id
            )
            if run.status == "requires_action":
                cur_tool_outputs = self._run_function_calling(run)
                sources.extend(cur_tool_outputs)

            time.sleep(self._run_retrieve_sleep_time)
        if run.status == "failed":
            raise ValueError(
                f"Run failed with status {run.status}.\n" f"Error: {run.last_error}"
            )
        return run, {"sources": sources}

    @property
    def latest_message(self) -> ChatMessage:
        """Get latest message."""
        raw_messages = self._client.beta.threads.messages.list(
            thread_id=self._thread_id, order="desc"
        )
        messages = from_openai_thread_messages(list(raw_messages))
        return messages[0]

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Main chat interface."""
        # TODO: since chat interface doesn't expose additional kwargs
        # we can't pass in file_ids per message
        added_message_obj = self.add_message(message)
        run, metadata = self.run_assistant(
            instructions_prefix=self._instructions_prefix,
        )
        latest_message = self.latest_message
        # get most recent message content
        return AgentChatResponse(
            response=str(latest_message.content),
            sources=metadata["sources"],
        )

    async def _achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        return self._chat(
            message,
            chat_history=chat_history,
            function_call=function_call,
            mode=mode,
        )

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
