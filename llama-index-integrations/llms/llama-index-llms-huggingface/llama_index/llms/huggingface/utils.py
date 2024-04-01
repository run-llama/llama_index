import requests
from packaging import version
from typing import Sequence, Union
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
)
from text_generation.types import (
    Message,
)


def resolve_tgi_function_call(url: str) -> bool:
    url = f"{url} + /info"
    model_info = dict(requests.get(url).json())
    tgi_version = model_info.get("version", None)
    if version.parse(tgi_version) >= version.parse("1.4.3"):
        return True
    else:
        raise ValueError(
            "'text-generation-inference' version ",
            f"incompatible with function call: {tgi_version}. ",
            "Function call support was added in v1.4.3",
        )


def get_max_input_length(url: str) -> Union[int, None]:
    url = f"{url} + /info"
    model_info = dict(requests.get(url).json())
    return model_info.get("max_input_length", None)


def to_tgi_messages(messages: Sequence[ChatMessage]) -> Sequence[Message]:
    messages = []
    for m in messages:
        tool_calls = m.additional_kwargs.get("tool_calls")
        messages.append(
            Message(role=m.role.value, content=m.content, tool_calls=tool_calls)
        )

    return messages


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


def resolve_tool_choice(tool_choice: Union[str, dict] = "auto") -> Union[str, dict]:
    """Resolve tool choice.

    If tool_choice is a function name string, return the appropriate dict.
    """
    if isinstance(tool_choice, str) and tool_choice not in ["none", "auto"]:
        return {"type": "function", "function": {"name": tool_choice}}

    return tool_choice
