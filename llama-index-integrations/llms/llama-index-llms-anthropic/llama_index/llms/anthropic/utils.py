"""
Utility functions for the Anthropic SDK LLM integration.
"""

from typing import Any, Dict, Sequence, Tuple, Optional

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ImageBlock,
    MessageRole,
    TextBlock,
    DocumentBlock,
)

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    DocumentBlockParam,
    ThinkingBlockParam,
    ImageBlockParam,
    CacheControlEphemeralParam,
    Base64PDFSourceParam,
)
from anthropic.types.tool_result_block_param import ToolResultBlockParam
from anthropic.types.tool_use_block_param import ToolUseBlockParam

HUMAN_PREFIX = "\n\nHuman:"
ASSISTANT_PREFIX = "\n\nAssistant:"

# AWS Bedrock Anthropic identifiers
BEDROCK_INFERENCE_PROFILE_CLAUDE_MODELS: Dict[str, int] = {
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-opus-20240229-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
    "anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
}
BEDROCK_CLAUDE_MODELS: Dict[str, int] = {
    "anthropic.claude-instant-v1": 100000,
    "anthropic.claude-v2": 100000,
    "anthropic.claude-v2:1": 200000,
}

# GCP Vertex AI Anthropic identifiers
VERTEX_CLAUDE_MODELS: Dict[str, int] = {
    "claude-3-opus@20240229": 200000,
    "claude-3-sonnet@20240229": 200000,
    "claude-3-haiku@20240307": 200000,
    "claude-3-5-sonnet@20240620": 200000,
    "claude-3-5-sonnet-v2@20241022": 200000,
    "claude-3-5-haiku@20241022": 200000,
    "claude-3-7-sonnet@20250219": 200000,
    "claude-opus-4@20250514": 200000,
    "claude-sonnet-4@20250514": 200000,
}

# Anthropic API/SDK identifiers
ANTHROPIC_MODELS: Dict[str, int] = {
    "claude-instant-1": 100000,
    "claude-instant-1.2": 100000,
    "claude-2": 100000,
    "claude-2.0": 100000,
    "claude-2.1": 200000,
    "claude-3-opus-latest": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-latest": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-latest": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-latest": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-latest": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-7-sonnet-20250219": 200000,
    "claude-3-7-sonnet-latest": 200000,
    "claude-opus-4-0": 200000,
    "claude-opus-4-20250514": 200000,
    "claude-4-opus-20250514": 200000,
    "claude-sonnet-4-0": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-4-sonnet-20250514": 200000,
}

# All provider Anthropic identifiers
CLAUDE_MODELS: Dict[str, int] = {
    **BEDROCK_INFERENCE_PROFILE_CLAUDE_MODELS,
    **BEDROCK_CLAUDE_MODELS,
    **VERTEX_CLAUDE_MODELS,
    **ANTHROPIC_MODELS,
}


def is_function_calling_model(modelname: str) -> bool:
    return "-3" in modelname or "-4" in modelname


def anthropic_modelname_to_contextsize(modelname: str) -> int:
    """
    Get the context size for an Anthropic model.

    Args:
        modelname (str): Anthropic model name.

    Returns:
        int: Context size for the specific model.

    """
    for model, context_size in BEDROCK_INFERENCE_PROFILE_CLAUDE_MODELS.items():
        # Only US & EU inference profiles are currently supported by AWS
        CLAUDE_MODELS[f"us.{model}"] = context_size
        CLAUDE_MODELS[f"eu.{model}"] = context_size

    if modelname not in CLAUDE_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Anthropic model name."
            "Known models are: " + ", ".join(CLAUDE_MODELS.keys())
        )

    return CLAUDE_MODELS[modelname]


def __merge_common_role_msgs(
    messages: Sequence[MessageParam],
) -> Sequence[MessageParam]:
    """Merge consecutive messages with the same role."""
    postprocessed_messages: Sequence[MessageParam] = []
    for message in messages:
        if (
            postprocessed_messages
            and postprocessed_messages[-1]["role"] == message["role"]
        ):
            postprocessed_messages[-1]["content"] += message["content"]
        else:
            postprocessed_messages.append(message)
    return postprocessed_messages


def messages_to_anthropic_messages(
    messages: Sequence[ChatMessage],
    cache_idx: Optional[int] = None,
) -> Tuple[Sequence[MessageParam], str]:
    """
    Converts a list of generic ChatMessages to anthropic messages.

    Args:
        messages: List of ChatMessages

    Returns:
        Tuple of:
        - List of anthropic messages
        - System prompt

    """
    anthropic_messages = []
    system_prompt = []
    for idx, message in enumerate(messages):
        # inject cache_control for all messages up to and including the cache_idx
        if cache_idx is not None and (idx <= cache_idx or cache_idx == -1):
            message.additional_kwargs["cache_control"] = {"type": "ephemeral"}

        if message.role == MessageRole.SYSTEM:
            for block in message.blocks:
                if isinstance(block, TextBlock) and block.text:
                    system_prompt.append(
                        _text_block_to_anthropic_message(
                            block, message.additional_kwargs
                        )
                    )
        elif message.role == MessageRole.FUNCTION or message.role == MessageRole.TOOL:
            content = ToolResultBlockParam(
                tool_use_id=message.additional_kwargs["tool_call_id"],
                type="tool_result",
                content=[TextBlockParam(text=message.content, type="text")],
                cache_control=message.additional_kwargs.get("cache_control"),
            )
            anth_message = MessageParam(
                role=MessageRole.USER.value,
                content=[content],
            )
            anthropic_messages.append(anth_message)
        else:
            content: list[TextBlockParam | ImageBlockParam | DocumentBlockParam] = []
            for block in message.blocks:
                if isinstance(block, TextBlock):
                    if block.text or message.additional_kwargs.get("thinking"):
                        content.append(
                            _text_block_to_anthropic_message(
                                block, message.additional_kwargs
                            )
                        )
                elif isinstance(block, ImageBlock):
                    # FUTURE: Claude does not support URLs, so we need to always convert to base64
                    img_bytes = block.resolve_image(as_base64=True).read()
                    img_str = img_bytes.decode("utf-8")

                    block_type = (
                        "document"
                        if block.image_mimetype == "application/pdf"
                        else "image"
                    )
                    block = ImageBlockParam(
                        type=block_type,
                        source={
                            "type": "base64",
                            "media_type": block.image_mimetype,
                            "data": img_str,
                        },
                    )
                    content.append(block)
                elif isinstance(block, DocumentBlock):
                    content.append(
                        _document_block_to_anthropic_message(
                            block=block, kwargs=message.additional_kwargs
                        )
                    )
            tool_calls = message.additional_kwargs.get("tool_calls", [])
            for tool_call in tool_calls:
                assert "id" in tool_call
                assert "input" in tool_call
                assert "name" in tool_call

                content.append(
                    ToolUseBlockParam(
                        id=tool_call["id"],
                        input=tool_call["input"],
                        name=tool_call["name"],
                        type="tool_use",
                    )
                )

            anth_message = MessageParam(
                role=message.role.value,
                content=content,
            )
            anthropic_messages.append(anth_message)

    return __merge_common_role_msgs(anthropic_messages), system_prompt


def _text_block_to_anthropic_message(
    block: TextBlock, kwargs: dict[str, Any]
) -> TextBlockParam:
    if "thinking" in kwargs and kwargs["thinking"] is not None:
        return ThinkingBlockParam(**kwargs["thinking"])
    elif "cache_control" in kwargs:
        return TextBlockParam(
            text=block.text,
            type="text",
            cache_control=CacheControlEphemeralParam(type="ephemeral"),
        )
    else:
        return TextBlockParam(text=block.text, type="text")


def _document_block_to_anthropic_message(
    block: DocumentBlock, kwargs: dict[str, Any]
) -> DocumentBlockParam:
    if not block.data:
        file_buffer = block.resolve_document()
        b64_string = block._get_b64_string(data_buffer=file_buffer)
    else:
        b64_string = block.data.decode("utf-8")
    return DocumentBlockParam(
        source=Base64PDFSourceParam(
            data=b64_string, media_type="application/pdf", type="base64"
        ),
        type="document",
    )


# Function used in bedrock
def _message_to_anthropic_prompt(message: ChatMessage) -> str:
    if message.role == MessageRole.USER:
        prompt = f"{HUMAN_PREFIX} {message.content}"
    elif message.role == MessageRole.ASSISTANT:
        prompt = f"{ASSISTANT_PREFIX} {message.content}"
    elif message.role == MessageRole.SYSTEM:
        prompt = f"{message.content}"
    elif message.role == MessageRole.FUNCTION:
        raise ValueError(f"Message role {MessageRole.FUNCTION} is not supported.")
    else:
        raise ValueError(f"Unknown message role: {message.role}")

    return prompt


def messages_to_anthropic_prompt(messages: Sequence[ChatMessage]) -> str:
    if len(messages) == 0:
        raise ValueError("Got empty list of messages.")

    # NOTE: make sure the prompt ends with the assistant prefix
    if messages[-1].role != MessageRole.ASSISTANT:
        messages = [
            *list(messages),
            ChatMessage(role=MessageRole.ASSISTANT, content=""),
        ]

    str_list = [_message_to_anthropic_prompt(message) for message in messages]
    return "".join(str_list)


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]
