import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from llama_index.core.base.llms.types import ChatMessage, LogProb
from packaging import version

MIN_ADS_VERSION = "2.12.6"

logger = logging.getLogger(__name__)


class UnsupportedOracleAdsVersionError(Exception):
    """
    Custom exception for unsupported `oracle-ads` versions.

    Attributes
    ----------
    current_version : str
        The installed version of `oracle-ads`.
    required_version : str
        The minimum required version of `oracle-ads`.
    """

    def __init__(self, current_version: str, required_version: str):
        super().__init__(
            f"The `oracle-ads` version {current_version} currently installed is incompatible with "
            "the `llama-index-llms-oci-data-science` version in use. To resolve this issue, "
            f"please upgrade to `oracle-ads:{required_version}` or later using the "
            "command: `pip install oracle-ads -U`"
        )


def _validate_dependency(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to validate the presence and version of the `oracle-ads` package.

    This decorator checks whether `oracle-ads` is installed and ensures its version meets
    the minimum requirement. Raises an error if the conditions are not met.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to wrap with the dependency validation.

    Returns
    -------
    Callable[..., Any]
        The wrapped function.

    Raises
    ------
    ImportError
        If `oracle-ads` is not installed.
    UnsupportedOracleAdsVersionError
        If the installed version is below the required version.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            from ads import __version__ as ads_version

            if version.parse(ads_version) < version.parse(MIN_ADS_VERSION):
                raise UnsupportedOracleAdsVersionError(ads_version, MIN_ADS_VERSION)

        except ImportError as ex:
            raise ImportError(
                "Could not import `oracle-ads` Python package. "
                "Please install it with `pip install oracle-ads`."
            ) from ex
        return func(*args, **kwargs)

    return wrapper


def _to_message_dicts(
    messages: Sequence[ChatMessage], drop_none: bool = False
) -> List[Dict[str, Any]]:
    """
    Converts a sequence of ChatMessage objects to a list of dictionaries.

    Parameters
    ----------
    messages : Sequence[ChatMessage]
        The messages to convert.
    drop_none : bool, optional
        Whether to drop keys with `None` values. Defaults to False.

    Returns
    -------
    List[Dict[str, Any]]
        The converted list of message dictionaries.
    """
    message_dicts = []
    for message in messages:
        message_dict = {
            "role": message.role.value,
            "content": message.content,
            **message.additional_kwargs,
        }
        if drop_none:
            message_dict = {k: v for k, v in message_dict.items() if v is not None}
        message_dicts.append(message_dict)
    return message_dicts


def _from_completion_logprobs_dict(
    completion_logprobs_dict: Dict[str, Any]
) -> List[List[LogProb]]:
    """
    Converts completion logprobs to a list of generic LogProb objects.

    Parameters
    ----------
    completion_logprobs_dict : Dict[str, Any]
        The completion logprobs to convert.

    Returns
    -------
    List[List[LogProb]]
        The converted logprobs.
    """
    return [
        [
            LogProb(token=token, logprob=logprob, bytes=[])
            for token, logprob in logprob_dict.items()
        ]
        for logprob_dict in completion_logprobs_dict.get("top_logprobs", [])
    ]


def _from_token_logprob_dicts(
    token_logprob_dicts: Sequence[Dict[str, Any]],
) -> List[List[LogProb]]:
    """
    Converts a sequence of token logprob dictionaries to a list of lists of LogProb objects.

    Parameters
    ----------
    token_logprob_dicts : Sequence[Dict[str, Any]]
        The token logprob dictionaries to convert.

    Returns
    -------
    List[List[LogProb]]
        The converted logprobs.
    """
    result = []
    for token_logprob_dict in token_logprob_dicts:
        try:
            logprobs_list = [
                LogProb(
                    token=el.get("token"),
                    logprob=el.get("logprob"),
                    bytes=el.get("bytes") or [],
                )
                for el in token_logprob_dict.get("top_logprobs", [])
            ]
            if logprobs_list:
                result.append(logprobs_list)
        except Exception as e:
            logger.warning(
                f"Error occurred in attempt to parse token logprob. "
                f"Details: {e}. Src: {token_logprob_dict}"
            )
    return result


def _from_message_dict(message_dict: Dict[str, Any]) -> ChatMessage:
    """
    Converts a message dictionary to a generic ChatMessage object.

    Parameters
    ----------
    message_dict : Dict[str, Any]
        The message dictionary.

    Returns
    -------
    ChatMessage
        The converted ChatMessage object.
    """
    role = message_dict.get("role")
    content = message_dict.get("content")
    additional_kwargs = {"tool_calls": message_dict.get("tool_calls", [])}
    return ChatMessage(role=role, content=content, additional_kwargs=additional_kwargs)


def _get_response_token_counts(raw_response: Dict[str, Any]) -> Dict[str, int]:
    """
    Extracts token usage information from the response.

    Parameters
    ----------
    raw_response : Dict[str, Any]
        The raw response containing token usage information.

    Returns
    -------
    Dict[str, int]
        The extracted token counts.
    """
    usage = raw_response.get("usage", {})

    if not usage:
        return {}

    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _update_tool_calls(
    tool_calls: List[Dict[str, Any]], tool_calls_delta: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Updates the tool calls using delta objects received from stream chunks.

    Parameters
    ----------
    tool_calls : List[Dict[str, Any]]
        The list of existing tool calls.
    tool_calls_delta : Optional[List[Dict[str, Any]]]
        The delta updates for the tool calls.

    Returns
    -------
    List[Dict[str, Any]]
        The updated tool calls.
    """
    if not tool_calls_delta:
        return tool_calls

    delta_call = tool_calls_delta[0]
    if not tool_calls or tool_calls[-1].get("index") != delta_call.get("index"):
        tool_calls.append(delta_call)
    else:
        latest_call = tool_calls[-1]
        latest_function = latest_call.setdefault("function", {})
        delta_function = delta_call.get("function", {})

        latest_function["arguments"] = latest_function.get(
            "arguments", ""
        ) + delta_function.get("arguments", "")
        latest_function["name"] = latest_function.get("name", "") + delta_function.get(
            "name", ""
        )
        latest_call["id"] = latest_call.get("id", "") + delta_call.get("id", "")

    return tool_calls


def _resolve_tool_choice(tool_choice: Union[str, dict] = "auto") -> Union[str, dict]:
    """Resolve tool choice.

    If tool_choice is a function name string, return the appropriate dict.
    """
    if isinstance(tool_choice, str) and tool_choice not in ["none", "auto", "required"]:
        return {"type": "function", "function": {"name": tool_choice}}

    return tool_choice
