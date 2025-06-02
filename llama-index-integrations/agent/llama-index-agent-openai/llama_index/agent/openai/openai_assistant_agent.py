"""OpenAI Assistant Agent."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from llama_index.core.agent.function_calling.step import (
    build_error_tool_output,
    build_missing_tool_message,
    get_function_by_name,
)
from llama_index.core.agent.types import BaseAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def from_openai_thread_message(thread_message: Any) -> ChatMessage:
    """From OpenAI thread message."""
    from openai.types.beta.threads import TextContentBlock, Message

    thread_message = cast(Message, thread_message)

    # we don't have a way of showing images, just do text for now
    text_contents = [
        t for t in thread_message.content if isinstance(t, TextContentBlock)
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
    if tool is not None:
        argument_dict = json.loads(arguments_str)
        output = tool(**argument_dict)

        if verbose:
            print(f"Got output: {output!s}")
            print("========================")
    else:
        err_msg = build_missing_tool_message(name)
        output = build_error_tool_output(name, arguments_str, err_msg)

        if verbose:
            print(err_msg)
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


async def acall_function(
    tools: List[BaseTool], fn_obj: Any, verbose: bool = False
) -> Tuple[ChatMessage, ToolOutput]:
    """Call an async function and return the output as a string."""
    from openai.types.beta.threads.required_action_function_tool_call import Function

    fn_obj = cast(Function, fn_obj)
    # TMP: consolidate with other abstractions
    name = fn_obj.name
    arguments_str = fn_obj.arguments

    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")

    tool = get_function_by_name(tools, name)
    if tool is not None:
        argument_dict = json.loads(arguments_str)
        tool = adapt_to_async_tool(tool)
        output = await tool.acall(**argument_dict)

        if verbose:
            print(f"Got output: {output!s}")
            print("========================")
    else:
        err_msg = build_missing_tool_message(name)
        output = build_error_tool_output(name, arguments_str, err_msg)

        if verbose:
            print(err_msg)
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


def _process_files(client: Any, files: List[str]) -> Dict[str, str]:
    """Process files."""
    from openai import OpenAI

    client = cast(OpenAI, client)

    file_dict = {}
    for file in files:
        file_obj = client.files.create(file=open(file, "rb"), purpose="assistants")
        file_dict[file_obj.id] = file
    return file_dict


def format_attachments(
    file_ids: Optional[List[str]] = None, tools: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Create attachments from file_ids and include tools."""
    file_ids = file_ids or []
    tools = tools or [{"type": "file_search"}]  # Default tool if none provided
    return [{"file_id": file_id, "tools": tools} for file_id in file_ids]


class OpenAIAssistantAgent(BaseAgent):
    """
    OpenAIAssistant agent.

    Wrapper around OpenAI assistant API: https://platform.openai.com/docs/assistants/overview

    """

    def __init__(
        self,
        client: Any,
        assistant: Any,
        tools: Optional[List[BaseTool]],
        callback_manager: Optional[CallbackManager] = None,
        thread_id: Optional[str] = None,
        instructions_prefix: Optional[str] = None,
        run_retrieve_sleep_time: float = 0.1,
        file_dict: Dict[str, str] = {},
        verbose: bool = False,
    ) -> None:
        """Init params."""
        from openai import OpenAI
        from openai.types.beta.assistant import Assistant

        self._client = cast(OpenAI, client)
        self._assistant = cast(Assistant, assistant)
        self._tools = tools or []
        if thread_id is None:
            thread = self._client.beta.threads.create()
            thread_id = thread.id
        self._thread_id = thread_id
        self._instructions_prefix = instructions_prefix
        self._run_retrieve_sleep_time = run_retrieve_sleep_time
        self._verbose = verbose
        self.file_dict = file_dict

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
        file_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> "OpenAIAssistantAgent":
        """
        From new assistant.

        Args:
            name: name of assistant
            instructions: instructions for assistant
            tools: list of tools
            openai_tools: list of openai tools
            thread_id: thread id
            model: model
            run_retrieve_sleep_time: run retrieve sleep time
            files: files
            instructions_prefix: instructions prefix
            callback_manager: callback manager
            verbose: verbose
            file_ids: list of file ids
            api_key: OpenAI API key
            top_p: model considers the results of the tokens with top_p probability mass.
            temperature: controls randomness of model

        """
        from openai import OpenAI

        # this is the set of openai tools
        # not to be confused with the tools we pass in for function calling
        openai_tools = openai_tools or []
        tools = tools or []
        tool_fns = [t.metadata.to_openai_tool() for t in tools]
        all_openai_tools = openai_tools + tool_fns

        # initialize client
        client = OpenAI(api_key=api_key)

        # process files
        files = files or []
        file_ids = file_ids or []

        file_dict = _process_files(client, files)

        # TODO: openai's typing is a bit sus
        all_openai_tools = cast(List[Any], all_openai_tools)
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=cast(List[Any], all_openai_tools),
            model=model,
            top_p=top_p,
            temperature=temperature,
        )
        return cls(
            client,
            assistant,
            tools,
            callback_manager=callback_manager,
            thread_id=thread_id,
            instructions_prefix=instructions_prefix,
            file_dict=file_dict,
            run_retrieve_sleep_time=run_retrieve_sleep_time,
            verbose=verbose,
        )

    @classmethod
    def from_existing(
        cls,
        assistant_id: str,
        tools: Optional[List[BaseTool]] = None,
        thread_id: Optional[str] = None,
        instructions_prefix: Optional[str] = None,
        run_retrieve_sleep_time: float = 0.1,
        callback_manager: Optional[CallbackManager] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ) -> "OpenAIAssistantAgent":
        """
        From existing assistant id.

        Args:
            assistant_id: id of assistant
            tools: list of BaseTools Assistant can use
            thread_id: thread id
            run_retrieve_sleep_time: run retrieve sleep time
            instructions_prefix: instructions prefix
            callback_manager: callback manager
            api_key: OpenAI API key
            verbose: verbose

        """
        from openai import OpenAI

        # initialize client
        client = OpenAI(api_key=api_key)

        # get assistant
        assistant = client.beta.assistants.retrieve(assistant_id)
        # assistant.tools is incompatible with BaseTools so have to pass from params

        return cls(
            client,
            assistant,
            tools=tools,
            callback_manager=callback_manager,
            thread_id=thread_id,
            instructions_prefix=instructions_prefix,
            run_retrieve_sleep_time=run_retrieve_sleep_time,
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
    def files_dict(self) -> Dict[str, str]:
        """Get files dict."""
        return self.file_dict

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

    def add_message(
        self,
        message: str,
        file_ids: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Add message to assistant."""
        attachments = format_attachments(file_ids=file_ids, tools=tools)
        return self._client.beta.threads.messages.create(
            thread_id=self._thread_id,
            role="user",
            content=message,
            attachments=attachments,
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

    async def _arun_function_calling(self, run: Any) -> List[ToolOutput]:
        """Run function calling."""
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tool_output_dicts = []
        tool_output_objs: List[ToolOutput] = []
        for tool_call in tool_calls:
            fn_obj = tool_call.function
            _, tool_output = await acall_function(
                self._tools, fn_obj, verbose=self._verbose
            )
            tool_output_dicts.append(
                {"tool_call_id": tool_call.id, "output": str(tool_output)}
            )
            tool_output_objs.append(tool_output)

        # submit tool outputs
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
            instructions=instructions_prefix,
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
                f"Run failed with status {run.status}.\nError: {run.last_error}"
            )
        return run, {"sources": sources}

    async def arun_assistant(
        self, instructions_prefix: Optional[str] = None
    ) -> Tuple[Any, Dict]:
        """Run assistant."""
        instructions_prefix = instructions_prefix or self._instructions_prefix
        run = self._client.beta.threads.runs.create(
            thread_id=self._thread_id,
            assistant_id=self._assistant.id,
            instructions=instructions_prefix,
        )
        from openai.types.beta.threads import Run

        run = cast(Run, run)

        sources = []

        while run.status in ["queued", "in_progress", "requires_action"]:
            run = self._client.beta.threads.runs.retrieve(
                thread_id=self._thread_id, run_id=run.id
            )
            if run.status == "requires_action":
                cur_tool_outputs = await self._arun_function_calling(run)
                sources.extend(cur_tool_outputs)

            await asyncio.sleep(self._run_retrieve_sleep_time)
        if run.status == "failed":
            raise ValueError(
                f"Run failed with status {run.status}.\nError: {run.last_error}"
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
        _added_message_obj = self.add_message(message)
        _run, metadata = self.run_assistant(
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
        """Asynchronous main chat interface."""
        self.add_message(message)
        run, metadata = await self.arun_assistant(
            instructions_prefix=self._instructions_prefix,
        )
        latest_message = self.latest_message
        # get most recent message content
        return AgentChatResponse(
            response=str(latest_message.content),
            sources=metadata["sources"],
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
