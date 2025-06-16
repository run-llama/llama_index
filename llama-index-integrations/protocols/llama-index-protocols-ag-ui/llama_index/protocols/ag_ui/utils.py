import datetime
import re
import uuid
from ag_ui.core import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    DeveloperMessage,
    ToolCall,
    FunctionCall,
)
from ag_ui.encoder import EventEncoder
from typing import Union, Callable

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.workflow import Event


def llama_index_message_to_ag_ui_message(
    message: ChatMessage,
) -> Message:
    msg_id = message.additional_kwargs.get("id", str(uuid.uuid4()))
    if message.role.value == "system":
        return SystemMessage(id=msg_id, content=message.content, role="system")
    elif (
        message.role.value == "user" and "tool_call_id" not in message.additional_kwargs
    ):
        message.content = re.sub(
            r"<state>[\s\S]*?</state>", "", message.content
        ).strip()
        return UserMessage(id=msg_id, content=message.content, role="user")
    elif message.role.value == "assistant":
        # Remove tool calls from the message
        if message.content:
            message.content = re.sub(
                r"<tool_call>[\s\S]*?</tool_call>", "", message.content
            ).strip()

        # Fetch tool calls from the message
        if message.additional_kwargs.get("ag_ui_tool_calls", None):
            tool_calls = [
                ToolCall(
                    type="function",
                    id=tool_call["id"],
                    function=FunctionCall(
                        name=tool_call["name"],
                        arguments=tool_call["arguments"],
                    ),
                )
                for tool_call in message.additional_kwargs["ag_ui_tool_calls"]
            ]
        else:
            tool_calls = None

        return AssistantMessage(
            id=msg_id,
            content=message.content or None,
            role="assistant",
            tool_calls=tool_calls,
        )
    elif message.role.value == "tool" or "tool_call_id" in message.additional_kwargs:
        return ToolMessage(
            id=msg_id,
            content=message.content or "",
            role="tool",
            tool_call_id=message.additional_kwargs.get(
                "tool_call_id", str(uuid.uuid4())
            ),
        )
    else:
        raise ValueError(f"Unknown message role: {message.role}")


def ag_ui_message_to_llama_index_message(message: Message) -> ChatMessage:
    if isinstance(message, SystemMessage):
        return ChatMessage(
            role="system", content=message.content, additional_kwargs={"id": message.id}
        )
    elif isinstance(message, UserMessage):
        return ChatMessage(
            role="user", content=message.content, additional_kwargs={"id": message.id}
        )
    elif isinstance(message, AssistantMessage):
        # TODO: llama-index-core needs to support tool calls on messages in a more official way
        # For now, we'll just convert the tool call into an assistant message
        # This is a bit of an opinionated hack for now to support the ag-ui tool calls
        tool_calls = message.tool_calls
        content = message.content or ""
        if tool_calls:
            tool_calls_str = "\n".join(
                [
                    f"<tool_call><name>{tool_call.function.name}</name><arguments>{tool_call.function.arguments}</arguments></tool_call>"
                    for tool_call in tool_calls
                ]
            )
            content = f"{content}\n\n{tool_calls_str}".strip()
            ag_ui_tool_calls = [
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
                for tool_call in tool_calls
            ]
        else:
            ag_ui_tool_calls = None

        return ChatMessage(
            role="assistant",
            content=content,
            additional_kwargs={
                "id": message.id,
                "ag_ui_tool_calls": ag_ui_tool_calls,
            },
        )
    elif isinstance(message, ToolMessage):
        # TODO: llama-index-core needs to support tool calls on messages in a more official way
        # tool call results into a user message
        # This is a bit of an opinionated hack for now to support the ag-ui tool calls
        return ChatMessage(
            role="user",
            content=message.content,
            additional_kwargs={"id": message.id, "tool_call_id": message.tool_call_id},
        )
    elif isinstance(message, DeveloperMessage):
        return ChatMessage(
            role="system", content=message.content, additional_kwargs={"id": message.id}
        )
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def workflow_event_to_sse(event: Event) -> str:
    event_str = EventEncoder().encode(event)

    print(event_str, flush=True)
    return event_str


def timestamp() -> int:
    return int(datetime.datetime.now().timestamp())


def validate_tool(tool: Union[BaseTool, Callable]) -> BaseTool:
    if isinstance(tool, BaseTool):
        return tool
    elif callable(tool):
        return FunctionTool.from_defaults(tool)
    else:
        raise ValueError(f"Invalid tool type: {type(tool)}")
