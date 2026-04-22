import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from llama_index.core.base.llms.types import ChatMessage, LogProb
from packaging import version

MIN_ADS_VERSION = "2.12.6"
SUPPORTED_TOOL_CHOICES = ["none", "auto", "required"]
DEFAULT_TOOL_CHOICE = "auto"

logger = logging.getLogger(__name__)


class UnsupportedOracleAdsVersionError(Exception):
    """
    Custom exception for unsupported `oracle-ads` versions.

    Attributes:
        current_version: The installed version of `oracle-ads`.
        required_version: The minimum required version of `oracle-ads`.

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
    Decorator to validate the presence and version of `oracle-ads`.

    This decorator checks that `oracle-ads` is installed and that its version meets
    the minimum requirement. If not, it raises an error.

    Args:
        func: The function to wrap with the dependency validation.

    Returns:
        The wrapped function.

    Raises:
        ImportError: If `oracle-ads` is not installed.
        UnsupportedOracleAdsVersionError: If the installed version is below the required version.

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
    Convert a sequence of ChatMessage objects to a list of dictionaries.

    Args:
        messages: The messages to convert.
        drop_none: Whether to drop keys with `None` values. Defaults to False.

    Returns:
        A list of message dictionaries.

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
    completion_logprobs_dict: Dict[str, Any],
) -> List[List[LogProb]]:
    """
    Convert completion logprobs to a list of generic LogProb objects.

    Args:
        completion_logprobs_dict: The completion logprobs to convert.

    Returns:
        A list of lists of LogProb objects.

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
    Convert a sequence of token logprob dictionaries to a list of LogProb objects.

    Args:
        token_logprob_dicts: The token logprob dictionaries to convert.

    Returns:
        A list of lists of LogProb objects.

    Raises:
        Warning: Logs a warning if an error occurs while parsing token logprobs.

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
                "Error occurred in attempt to parse token logprob. "
                f"Details: {e}. Src: {token_logprob_dict}"
            )
    return result


def _from_message_dict(message_dict: Dict[str, Any]) -> ChatMessage:
    """
    Convert a message dictionary to a ChatMessage object.

    Args:
        message_dict: The message dictionary.

    Returns:
        A ChatMessage object representing the given dictionary.

    """
    return ChatMessage(
        role=message_dict.get("role"),
        content=message_dict.get("content"),
        additional_kwargs={"tool_calls": message_dict.get("tool_calls", [])},
    )


def _get_response_token_counts(raw_response: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract token usage information from the response.

    Args:
        raw_response: The raw response containing token usage information.

    Returns:
        A dictionary containing token counts, or an empty dictionary if usage info is not found.

    """
    if not raw_response.get("usage"):
        return {}

    return {
        "prompt_tokens": raw_response["usage"].get("prompt_tokens", 0),
        "completion_tokens": raw_response["usage"].get("completion_tokens", 0),
        "total_tokens": raw_response["usage"].get("total_tokens", 0),
    }


def _update_tool_calls(
    tool_calls: List[Dict[str, Any]], tool_calls_delta: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Update the tool calls using delta objects received from stream chunks.

    Args:
        tool_calls: The list of existing tool calls.
        tool_calls_delta: The delta updates for the tool calls (if any).

    Returns:
        The updated list of tool calls.

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


def _resolve_tool_choice(
    tool_choice: Optional[Union[str, dict]] = None,
    tool_required: bool = False,
) -> Union[str, dict]:
    """
    Resolve the tool choice into a string or a dictionary.

    If the tool_choice is a string that is not in SUPPORTED_TOOL_CHOICES, a dictionary
    representing a function call is returned.

    Args:
        tool_choice: The desired tool choice, which can be a string or a dictionary. Defaults to "auto".

    Returns:
        Either the original tool_choice if valid or a dictionary representing a function call.

    """
    if tool_choice is None:
        tool_choice = "required" if tool_required else "auto"
    if isinstance(tool_choice, str) and tool_choice not in SUPPORTED_TOOL_CHOICES:
        return {"type": "function", "function": {"name": tool_choice}}
    return tool_choice


_KNOWN_SUFFIXES = ("/predict", "/predictWithStreamResponse")


def _strip_md_endpoint_suffix(endpoint: str) -> str:
    """
    Normalize a user-provided OCI Model Deployment endpoint into its base form.

    Users may configure the endpoint in multiple valid ways, for example:

        https://<MD_OCID>
        https://<MD_OCID>/predict
        https://<MD_OCID>/predictWithStreamResponse

    This helper removes any known suffix ("/predict" or
    "/predictWithStreamResponse") and returns only the base endpoint:

        https://<MD_OCID>

    This ensures downstream URL construction is consistent and avoids
    incorrect paths such as:

        .../predict/predictWithStreamResponse

    Args:
        endpoint (str): User-provided endpoint URL.

    Returns:
        str: Base endpoint URL without any known prediction suffix.

    """
    ep = (endpoint or "").rstrip("/")
    for sfx in _KNOWN_SUFFIXES:
        if ep.endswith(sfx):
            return ep[: -len(sfx)]
    return ep


def _endpoint_suffix(endpoint: str) -> str:
    """
    Detect which known suffix is present in the configured endpoint.

    This function inspects the endpoint string and returns:

        - "predict" if the endpoint ends with "/predict"
        - "predictWithStreamResponse" if it ends with "/predictWithStreamResponse"
        - "" if no known suffix is present (base-only endpoint)

    This allows the client to infer whether the user explicitly configured
    v1 streaming (/predict) or v2 streaming (/predictWithStreamResponse).

    Args:
        endpoint (str): User-provided endpoint URL.

    Returns:
        str: The detected suffix name, or an empty string if none is found.

    """
    ep = (endpoint or "").rstrip("/")
    if ep.endswith("/predictWithStreamResponse"):
        return "predictWithStreamResponse"
    if ep.endswith("/predict"):
        return "predict"
    return ""


def _resolve_invoke_url(endpoint: str, *, stream: bool) -> str:
    """
    Resolve the final URL to invoke for a given request type (streaming vs non-streaming).

    OCI Data Science Model Deployments support:

    - Standard request/response inference:
        POST <base>/predict

    - Legacy streaming inference (v1):
        POST <base>/predict   with stream=True and SSE-style response chunks

    - GA streaming inference (v2):
        POST <base>/predictWithStreamResponse

    Users may configure the client endpoint in any of these forms:

        https://<MD_OCID>
        https://<MD_OCID>/predict
        https://<MD_OCID>/predictWithStreamResponse

    This resolver ensures:

    Non-streaming calls:
        Always go to <base>/predict

    Streaming calls:
        - Use /predict if the user configured endpoint with "/predict"
        - Use /predictWithStreamResponse if user configured that endpoint
        - If user provided only the base endpoint, streaming defaults to v2

    This guarantees backward compatibility while allowing GA streaming support,
    without mutating the stored endpoint configuration.

    Args:
        endpoint (str): User-configured endpoint URL.
        stream (bool): Whether the current request is a streaming request.

    Returns:
        str: The resolved URL that should be invoked for this request.

    """
    base = _strip_md_endpoint_suffix(endpoint)

    if not stream:
        return f"{base}/predict"

    suffix = _endpoint_suffix(endpoint)
    if suffix == "predict":
        return f"{base}/predict"
    if suffix == "predictWithStreamResponse":
        return f"{base}/predictWithStreamResponse"

    # base only => default streaming to v2
    return f"{base}/predictWithStreamResponse"
