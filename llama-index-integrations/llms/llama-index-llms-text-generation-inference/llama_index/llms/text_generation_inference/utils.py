import requests
from packaging import version
from typing import Sequence, Union, List, Optional
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
)
from text_generation.types import (
    Message,
)


def resolve_tgi_function_call(url: str) -> bool:
    url = f"{url}/info"
    model_info = dict(requests.get(url).json())
    tgi_version = model_info.get("version", None)
    if version.parse(tgi_version) >= version.parse("2.0.1"):
        return True
    else:
        raise ValueError(
            "'text-generation-inference' version ",
            f"incompatible with function call: {tgi_version}. ",
            "Function call support was added in v2.0.1",
        )


def get_max_input_length(url: str) -> Union[int, None]:
    url = f"{url}/info"
    model_info = dict(requests.get(url).json())
    return model_info.get("max_input_length", None)


def to_tgi_messages(messages: Sequence[ChatMessage]) -> Sequence[Message]:
    out_messages = []
    for m in messages:
        tool_calls = m.additional_kwargs.get("tool_calls")
        out_messages.append(
            Message(role=m.role.value, content=m.content, tool_calls=tool_calls)
        )

    return out_messages


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


def resolve_tool_choice(
    tools: Optional[List[dict]] = None, tool_choice: str = "none"
) -> Union[str, dict]:
    """Resolve tool choice.

    Check if tool_name exists in tools.
    Note that unlike in OpenAI specification, 'auto' will ALWAYS choose the tool for you.
    Set to 'none' explicitly if do not wish to use tool.
    """
    valid_tool_choices = ["none", "auto"] + [t["function"]["name"] for t in tools or []]

    if tool_choice not in valid_tool_choices:
        raise ValueError(
            f"{tool_choice} is not a valid tool_choice. Must be one of {valid_tool_choices}"
        )

    return tool_choice
