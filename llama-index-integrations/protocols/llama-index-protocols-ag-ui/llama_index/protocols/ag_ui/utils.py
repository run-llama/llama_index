import datetime
import json
import uuid
from ag_ui.core import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    DeveloperMessage,
)

from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Event


def llama_index_message_to_ag_ui_message(message: ChatMessage) -> Message:
    msg_id = message.additional_kwargs.get("id", str(uuid.uuid4()))
    if message.role == "system":
        return SystemMessage(id=msg_id, content=message.content)
    elif message.role == "user":
        return UserMessage(id=msg_id, content=message.content)
    elif message.role == "assistant":
        return AssistantMessage(id=msg_id, content=message.content)
    elif message.role == "tool":
        return ToolMessage(id=msg_id, content=message.content)
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

        return ChatMessage(
            role="assistant", content=content, additional_kwargs={"id": message.id}
        )
    elif isinstance(message, ToolMessage):
        # TODO: llama-index-core needs to support tool calls on messages in a more official way
        # tool call results into a user message
        # This is a bit of an opinionated hack for now to support the ag-ui tool calls
        return ChatMessage(
            role="user", content=message.content, additional_kwargs={"id": message.id}
        )
    elif isinstance(message, DeveloperMessage):
        return ChatMessage(
            role="system", content=message.content, additional_kwargs={"id": message.id}
        )
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def workflow_event_to_sse(event: Event) -> str:
    event_dict = event.model_dump()

    # TODO: ag-ui SDK does not have the proper field names
    # Need to convert snake_case to camelCase
    proper_dict = {}
    for key, value in event_dict.items():
        key_parts = key.split("_")
        new_key = key_parts[0] + "".join(word.capitalize() for word in key_parts[1:])
        proper_dict[new_key] = value

    print(f"data: {json.dumps(proper_dict, ensure_ascii=False)}\n\n", flush=True)
    return f"data: {json.dumps(proper_dict, ensure_ascii=False)}\n\n"


def timestamp() -> int:
    return int(datetime.datetime.now().timestamp())


def get_kwargs_delta(new_json: str, old_json: str) -> str:
    """
    Convert complete JSON to incremental fragments by finding the minimal
    string that needs to be appended to old_json to get new_json.
    """
    # Handle the first call - send everything except closing brace
    if not old_json:
        if new_json.endswith("}"):
            return new_json[:-1]
        return new_json

    # If nothing changed, return empty
    if new_json == old_json:
        return ""

    # Remove closing braces/quotes for comparison
    old_working = old_json.rstrip('"}')
    new_working = new_json.rstrip('"}')

    # If the new content starts with what we had before, we can append
    if new_working.startswith(old_working):
        return new_working[len(old_working) :]

    # If there's a mismatch (like null values being replaced),
    # we need to find the longest common prefix and send the difference
    common_length = 0
    min_length = min(len(old_working), len(new_working))

    for i in range(min_length):
        if new_working[i] == old_working[i]:
            common_length += 1
        else:
            break

    # Return what needs to be added after the common part
    return new_working[common_length:]
