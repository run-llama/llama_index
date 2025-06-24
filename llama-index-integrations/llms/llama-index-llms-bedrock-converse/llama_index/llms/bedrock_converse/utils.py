import base64
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    ImageBlock,
    TextBlock,
    ContentBlock,
    AudioBlock,
    DocumentBlock,
)


logger = logging.getLogger(__name__)

HUMAN_PREFIX = "\n\nHuman:"
ASSISTANT_PREFIX = "\n\nAssistant:"

BEDROCK_MODELS = {
    "amazon.nova-pro-v1:0": 300000,
    "amazon.nova-lite-v1:0": 300000,
    "amazon.nova-micro-v1:0": 128000,
    "amazon.titan-text-express-v1": 8192,
    "amazon.titan-text-lite-v1": 4096,
    "amazon.titan-text-premier-v1:0": 3072,
    "anthropic.claude-instant-v1": 100000,
    "anthropic.claude-v1": 100000,
    "anthropic.claude-v2": 100000,
    "anthropic.claude-v2:1": 200000,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "anthropic.claude-3-opus-20240229-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
    "anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "ai21.j2-mid-v1": 8192,
    "ai21.j2-ultra-v1": 8192,
    "cohere.command-text-v14": 4096,
    "cohere.command-light-text-v14": 4096,
    "cohere.command-r-v1:0": 128000,
    "cohere.command-r-plus-v1:0": 128000,
    "meta.llama2-13b-chat-v1": 2048,
    "meta.llama2-70b-chat-v1": 4096,
    "meta.llama3-8b-instruct-v1:0": 8192,
    "meta.llama3-70b-instruct-v1:0": 8192,
    "meta.llama3-1-8b-instruct-v1:0": 128000,
    "meta.llama3-1-70b-instruct-v1:0": 128000,
    "meta.llama3-2-1b-instruct-v1:0": 131000,
    "meta.llama3-2-3b-instruct-v1:0": 131000,
    "meta.llama3-2-11b-instruct-v1:0": 128000,
    "meta.llama3-2-90b-instruct-v1:0": 128000,
    "meta.llama3-3-70b-instruct-v1:0": 128000,
    "mistral.mistral-7b-instruct-v0:2": 32000,
    "mistral.mixtral-8x7b-instruct-v0:1": 32000,
    "mistral.mistral-large-2402-v1:0": 32000,
    "mistral.mistral-small-2402-v1:0": 32000,
    "mistral.mistral-large-2407-v1:0": 32000,
    "ai21.jamba-1-5-mini-v1:0": 256000,
    "ai21.jamba-1-5-large-v1:0": 256000,
    "deepseek.r1-v1:0": 128000,
}

BEDROCK_FUNCTION_CALLING_MODELS = (
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "cohere.command-r-v1:0",
    "cohere.command-r-plus-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mistral-large-2407-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-2-1b-instruct-v1:0",
    "meta.llama3-2-3b-instruct-v1:0",
    "meta.llama3-2-11b-instruct-v1:0",
    "meta.llama3-2-90b-instruct-v1:0",
    "meta.llama3-3-70b-instruct-v1:0",
)

BEDROCK_INFERENCE_PROFILE_SUPPORTED_MODELS = (
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-2-1b-instruct-v1:0",
    "meta.llama3-2-3b-instruct-v1:0",
    "meta.llama3-2-11b-instruct-v1:0",
    "meta.llama3-2-90b-instruct-v1:0",
    "meta.llama3-3-70b-instruct-v1:0",
    "deepseek.r1-v1:0",
)


def get_model_name(model_name: str) -> str:
    """Extract base model name from region-prefixed model identifier."""
    # Check for region prefixes (us, eu, apac)
    REGION_PREFIXES = ["us.", "eu.", "apac."]

    # If no region prefix, return the original model name
    if not any(prefix in model_name for prefix in REGION_PREFIXES):
        return model_name

    # Remove region prefix to get the base model name
    base_model_name = model_name[model_name.find(".") + 1 :]

    if base_model_name not in BEDROCK_INFERENCE_PROFILE_SUPPORTED_MODELS:
        raise ValueError(
            f"Model does not support inference profiles but has an inference profile prefix: {model_name}. "
            "Please provide a valid Bedrock model name. "
            "Known models are: " + ", ".join(BEDROCK_INFERENCE_PROFILE_SUPPORTED_MODELS)
        )

    return base_model_name


def is_bedrock_function_calling_model(model_name: str) -> bool:
    return get_model_name(model_name) in BEDROCK_FUNCTION_CALLING_MODELS


def bedrock_modelname_to_context_size(model_name: str) -> int:
    translated_model_name = get_model_name(model_name)

    if translated_model_name not in BEDROCK_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid Bedrock model name. "
            "Known models are: " + ", ".join(BEDROCK_MODELS.keys())
        )

    return BEDROCK_MODELS[translated_model_name]


def __merge_common_role_msgs(
    messages: Sequence[Dict[str, Any]],
) -> Sequence[Dict[str, Any]]:
    """Merge consecutive messages with the same role."""
    postprocessed_messages: Sequence[Dict[str, Any]] = []
    for message in messages:
        if (
            postprocessed_messages
            and postprocessed_messages[-1]["role"] == message["role"]
        ):
            postprocessed_messages[-1]["content"] += message["content"]
        else:
            postprocessed_messages.append(message)
    return postprocessed_messages


def _content_block_to_bedrock_format(
    block: ContentBlock, role: MessageRole
) -> Optional[Dict[str, Any]]:
    """Convert content block to AWS Bedrock Converse API required format."""
    if isinstance(block, TextBlock):
        if not block.text:
            return None
        return {
            "text": block.text,
        }
    elif isinstance(block, DocumentBlock):
        if not block.data:
            file_buffer = block.resolve_document()
            with file_buffer as f:
                data = f.read()
        else:
            data = base64.b64decode(block.data)
        title = block.title
        # NOTE: At the time of writing, "txt" format works for all file types
        # The API then infers the format from the file type based on the bytes
        return {"document": {"format": "txt", "name": title, "source": {"bytes": data}}}
    elif isinstance(block, ImageBlock):
        if role != MessageRole.USER:
            logger.warning(
                "Bedrock Converse API only supports image blocks for user messages."
            )
            return None
        # NOTE: Bedrock Converse API accepts raw image bytes directly
        # No need to convert the image to base64 or UTF-8 format
        # The image will be passed to the API in its raw binary form
        img_format = __get_img_format_from_image_mimetype(block.image_mimetype)
        raw_image_bytes = block.resolve_image(as_base64=False).read()
        return {"image": {"format": img_format, "source": {"bytes": raw_image_bytes}}}
    elif isinstance(block, AudioBlock):
        logger.warning("Audio blocks are not supported in Bedrock Converse API.")
        return None
    else:
        logger.warning(f"Unsupported block type: {type(block)}")
        return None


def __get_img_format_from_image_mimetype(image_mimetype: str) -> str:
    """Get the image format from the image mimetype."""
    mapping = {
        "image/jpeg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    img_format = mapping.get(image_mimetype)
    if img_format is None:
        logger.warning(
            f"Unsupported image mimetype: {image_mimetype}. Bedrock Converse API only supports {', '.join(mapping.keys())}, defaulting to png."
        )
        return "png"
    return img_format


def messages_to_converse_messages(
    messages: Sequence[ChatMessage],
) -> Tuple[Sequence[Dict[str, Any]], str]:
    """
    Converts a list of generic ChatMessages to AWS Bedrock Converse messages.

    Args:
        messages: List of ChatMessages

    Returns:
        Tuple of:
        - List of AWS Bedrock Converse messages
        - System prompt

    """
    converse_messages = []
    system_prompt = ""
    for message in messages:
        if message.role == MessageRole.SYSTEM and message.content:
            system_prompt += (message.content) + "\n"
        elif message.role in [MessageRole.FUNCTION, MessageRole.TOOL]:
            # convert tool output to the AWS Bedrock Converse format
            content = {
                "toolResult": {
                    "toolUseId": message.additional_kwargs["tool_call_id"],
                    "content": [{"text": message.content}] if message.content else [],
                }
            }
            if status := message.additional_kwargs.get("status"):
                content["toolResult"]["status"] = status
            converse_messages.append(
                {
                    "role": MessageRole.USER.value,  # bedrock converse api requires the role to be user
                    "content": [content],
                }
            )
        else:
            content = []
            for block in message.blocks:
                bedrock_format_block = _content_block_to_bedrock_format(
                    block=block,
                    role=message.role,
                )
                if bedrock_format_block:
                    content.append(bedrock_format_block)

            if content:
                converse_messages.append(
                    {
                        "role": message.role.value,
                        "content": content,
                    }
                )

        # convert tool calls to the AWS Bedrock Converse format
        # NOTE tool calls might show up within any message,
        # e.g. within assistant message or in consecutive tool calls,
        # thus this tool call check is done for all messages
        tool_calls = message.additional_kwargs.get("tool_calls", [])
        content = []
        for tool_call in tool_calls:
            assert "toolUseId" in tool_call, f"`toolUseId` not found in {tool_call}"
            assert "input" in tool_call, f"`input` not found in {tool_call}"
            assert "name" in tool_call, f"`name` not found in {tool_call}"
            tool_input = tool_call["input"] if tool_call["input"] else {}
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input or "{}")
                except json.JSONDecodeError:
                    tool_input = {}

            content.append(
                {
                    "toolUse": {
                        "input": tool_input,
                        "toolUseId": tool_call["toolUseId"],
                        "name": tool_call["name"],
                    }
                }
            )
        if len(content) > 0:
            converse_messages.append(
                {
                    "role": "assistant",  # tool calls are always from the assistant
                    "content": content,
                }
            )

    return __merge_common_role_msgs(converse_messages), system_prompt.strip()


def tools_to_converse_tools(
    tools: List["BaseTool"],
    tool_choice: Optional[dict] = None,
    tool_required: bool = False,
) -> Dict[str, Any]:
    """
    Converts a list of tools to AWS Bedrock Converse tools.

    Args:
        tools: List of BaseTools

    Returns:
        AWS Bedrock Converse tools

    """
    converse_tools = []
    for tool in tools:
        tool_name, tool_description = tool.metadata.name, tool.metadata.description
        if not tool_name:
            raise ValueError(f"Tool {tool} does not have a name.")

        tool_dict = {
            "name": tool_name,
            "description": tool_description,
            # get the schema of the tool's input parameters in the format expected by AWS Bedrock Converse
            "inputSchema": {"json": tool.metadata.get_parameters_dict()},
        }
        converse_tools.append({"toolSpec": tool_dict})
    return {
        "tools": converse_tools,
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
        # e.g. { "auto": {} }
        "toolChoice": tool_choice or ({"any": {}} if tool_required else {"auto": {}}),
    }


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


def _create_retry_decorator(client: Any, max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    try:
        import boto3  # noqa
    except ImportError as e:
        raise ImportError(
            "You must install the `boto3` package to use Bedrock."
            "Please `pip install boto3`"
        ) from e

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(client.exceptions.ThrottlingException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _create_retry_decorator_async(max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    try:
        import aioboto3  # noqa
    except ImportError as e:
        raise ImportError(
            "You must install the `aioboto3` package to use Bedrock."
            "Please `pip install aioboto3`"
        ) from e

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type()
        ),  # TODO: Add throttling exception in async version
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def converse_with_retry(
    client: Any,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_retries: int = 3,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.1,
    stream: bool = False,
    guardrail_identifier: Optional[str] = None,
    guardrail_version: Optional[str] = None,
    trace: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(client=client, max_retries=max_retries)
    converse_kwargs = {
        "modelId": model,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system_prompt:
        converse_kwargs["system"] = [{"text": system_prompt}]
    if tool_config := kwargs.get("tools"):
        converse_kwargs["toolConfig"] = tool_config
    if guardrail_identifier and guardrail_version:
        converse_kwargs["guardrailConfig"] = {}
        converse_kwargs["guardrailConfig"]["guardrailIdentifier"] = guardrail_identifier
        converse_kwargs["guardrailConfig"]["guardrailVersion"] = guardrail_version
        if trace:
            converse_kwargs["guardrailConfig"]["trace"] = trace
    converse_kwargs = join_two_dicts(
        converse_kwargs,
        {
            k: v
            for k, v in kwargs.items()
            if k not in ["tools", "guardrail_identifier", "guardrail_version", "trace"]
        },
    )

    @retry_decorator
    def _conversion_with_retry(**kwargs: Any) -> Any:
        if stream:
            return client.converse_stream(**kwargs)
        return client.converse(**kwargs)

    return _conversion_with_retry(**converse_kwargs)


async def converse_with_retry_async(
    session: Any,
    config: Any,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_retries: int = 3,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.1,
    stream: bool = False,
    guardrail_identifier: Optional[str] = None,
    guardrail_version: Optional[str] = None,
    trace: Optional[str] = None,
    boto_client_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator_async(max_retries=max_retries)
    converse_kwargs = {
        "modelId": model,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system_prompt:
        converse_kwargs["system"] = [{"text": system_prompt}]
    if tool_config := kwargs.get("tools"):
        converse_kwargs["toolConfig"] = tool_config
    if guardrail_identifier and guardrail_version:
        converse_kwargs["guardrailConfig"] = {}
        converse_kwargs["guardrailConfig"]["guardrailIdentifier"] = guardrail_identifier
        converse_kwargs["guardrailConfig"]["guardrailVersion"] = guardrail_version
        if trace:
            converse_kwargs["guardrailConfig"]["trace"] = trace
    converse_kwargs = join_two_dicts(
        converse_kwargs,
        {
            k: v
            for k, v in kwargs.items()
            if k not in ["tools", "guardrail_identifier", "guardrail_version", "trace"]
        },
    )
    _boto_client_kwargs = {}
    if boto_client_kwargs is not None:
        _boto_client_kwargs |= boto_client_kwargs

    ## NOTE: Returning the generator directly from converse_stream doesn't work
    # So, we have to use two separate functions for streaming and non-streaming
    # This differs from the synchronous version, and is a bit of a hack
    # Further investigation is needed

    @retry_decorator
    async def _conversion_with_retry(**kwargs: Any) -> Any:
        async with session.client(
            "bedrock-runtime",
            config=config,
            **_boto_client_kwargs,
        ) as client:
            return await client.converse(**kwargs)

    @retry_decorator
    async def _conversion_stream_with_retry(**kwargs: Any) -> Any:
        async with session.client(
            "bedrock-runtime",
            config=config,
            **_boto_client_kwargs,
        ) as client:
            response = await client.converse_stream(**kwargs)
            async for event in response["stream"]:
                yield event

    if stream:
        return _conversion_stream_with_retry(**converse_kwargs)
    else:
        return await _conversion_with_retry(**converse_kwargs)


def join_two_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Joins two dictionaries, summing shared keys and adding new keys.

    Args:
        dict1: First dictionary
        dict2: Second dictionary

    Returns:
        Joined dictionary

    """
    # These keys should be replaced rather than concatenated
    REPLACE_KEYS = {"toolUseId", "name", "input"}

    new_dict = dict1.copy()
    for key, value in dict2.items():
        if key not in new_dict:
            new_dict[key] = value
        else:
            if isinstance(value, dict):
                new_dict[key] = join_two_dicts(new_dict[key], value)
            elif key in REPLACE_KEYS:
                # Replace instead of concatenate for special keys
                new_dict[key] = value
            else:
                new_dict[key] += value
    return new_dict
