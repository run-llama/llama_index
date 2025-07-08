import requests
from packaging import version
from typing import Union, List, Optional
from llama_index.core.base.llms.types import (
    ChatResponse,
)


def get_max_input_tokens(url: str) -> Union[int, None]:
    url = f"{url}/info"
    model_info = dict(requests.get(url).json())
    tgi_version = model_info.get("version")
    if version.parse(tgi_version) >= version.parse("2.1.0"):
        return model_info.get("max_input_tokens")
    else:
        return model_info.get("max_input_length")


def get_max_total_tokens(url: str) -> Union[int, None]:
    url = f"{url}/info"
    model_info = dict(requests.get(url).json())
    return model_info.get("max_total_tokens")


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


def resolve_tool_choice(
    tools: Optional[List[dict]] = None, tool_choice: str = "none"
) -> Union[str, dict]:
    """
    Resolve tool choice.

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
