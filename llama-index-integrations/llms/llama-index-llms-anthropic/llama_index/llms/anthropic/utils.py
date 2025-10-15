"""
Utility functions for the Anthropic SDK LLM integration.
"""

from typing import Any, Dict, List, Sequence, Tuple, Optional

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ImageBlock,
    MessageRole,
    TextBlock,
    DocumentBlock,
    CachePoint,
    CitableBlock,
    CitationBlock,
    ThinkingBlock,
    ContentBlock,
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
from anthropic.types import ContentBlockParam as AnthropicContentBlock
from anthropic.types.beta import (
    BetaSearchResultBlockParam,
    BetaTextBlockParam,
    BetaCitationsConfigParam,
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
    "anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-sonnet-4-20250514-v1:0": 1000000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
    "anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
    "anthropic.claude-haiku-4-5-20251001-v1:0": 200000,
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
    "claude-opus-4-1@20250805": 200000,
    "claude-sonnet-4-5@20250929": 200000,
    "claude-haiku-4-5@20251001": 200000,
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
    "claude-sonnet-4-0": 1000000,
    "claude-sonnet-4-20250514": 1000000,
    "claude-4-sonnet-20250514": 1000000,
    "claude-opus-4-1-20250805": 200000,
    "claude-opus-4-1": 200000,
    "claude-sonnet-4-5-20250929": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-haiku-4-5": 200000,
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


def _to_anthropic_text_block(block: TextBlock) -> TextBlockParam:
    return TextBlockParam(
        type="text",
        text=block.text,
    )


def _to_anthropic_image_block(block: ImageBlock) -> ImageBlockParam:
    # FUTURE: Claude does not support URLs, so we need to always convert to base64
    img_bytes = block.resolve_image(as_base64=True).read()
    img_str = img_bytes.decode("utf-8")

    block_type = "document" if block.image_mimetype == "application/pdf" else "image"
    return ImageBlockParam(
        type=block_type,
        source={
            "type": "base64",
            "media_type": block.image_mimetype,
            "data": img_str,
        },
    )


def _to_anthropic_document_block(block: DocumentBlock) -> DocumentBlockParam:
    if not block.data:
        file_buffer = block.resolve_document()
        b64_string = block._get_b64_string(data_buffer=file_buffer)
    else:
        b64_string = block.data.decode("utf-8")

    return DocumentBlockParam(
        type="document",
        source=Base64PDFSourceParam(
            data=b64_string, media_type="application/pdf", type="base64"
        ),
    )


def blocks_to_anthropic_blocks(
    blocks: Sequence[ContentBlock], kwargs: dict[str, Any]
) -> List[AnthropicContentBlock]:
    anthropic_blocks: List[AnthropicContentBlock] = []
    global_cache_control: Optional[CacheControlEphemeralParam] = None

    if kwargs.get("cache_control"):
        global_cache_control = CacheControlEphemeralParam(**kwargs["cache_control"])

    for block in blocks:
        if isinstance(block, TextBlock):
            if block.text:
                anthropic_blocks.append(_to_anthropic_text_block(block))
                if global_cache_control:
                    anthropic_blocks[-1]["cache_control"] = global_cache_control

        elif isinstance(block, ImageBlock):
            anthropic_blocks.append(_to_anthropic_image_block(block))
            if global_cache_control:
                anthropic_blocks[-1]["cache_control"] = global_cache_control

        elif isinstance(block, DocumentBlock):
            anthropic_blocks.append(_to_anthropic_document_block(block))
            if global_cache_control:
                anthropic_blocks[-1]["cache_control"] = global_cache_control

        elif isinstance(block, CitableBlock):
            anthropic_sub_blocks = []
            for sub_block in block.content:
                if isinstance(sub_block, TextBlock) and sub_block.text:
                    anthropic_sub_blocks.append(
                        BetaTextBlockParam(
                            type="text",
                            text=sub_block.text,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported block type for citable blocks: {type(sub_block)}"
                    )

            anthropic_blocks.append(
                BetaSearchResultBlockParam(
                    type="search_result",
                    content=anthropic_sub_blocks,
                    source=block.source,
                    title=block.title,
                    cache_control=global_cache_control,
                    citations=BetaCitationsConfigParam(
                        enabled=True,
                    ),
                )
            )
            if global_cache_control:
                anthropic_blocks[-1]["cache_control"] = global_cache_control

        elif isinstance(block, ThinkingBlock):
            if block.content:
                signature = block.additional_information.get("signature", "")
                anthropic_blocks.append(
                    ThinkingBlockParam(
                        signature=signature, thinking=block.content, type="thinking"
                    )
                )
                if global_cache_control:
                    anthropic_blocks[-1]["cache_control"] = global_cache_control

        elif isinstance(block, CachePoint):
            if len(anthropic_blocks) > 0:
                anthropic_blocks[-1]["cache_control"] = CacheControlEphemeralParam(
                    **block.cache_control.model_dump()
                )
            else:
                raise ValueError("Cache point must be after at least one block")
        elif isinstance(block, CitationBlock):
            # No need to pass these back to Anthropic
            continue
        else:
            raise ValueError(f"Unsupported block type: {type(block)}")

    tool_calls = kwargs.get("tool_calls", [])
    for tool_call in tool_calls:
        assert "id" in tool_call
        assert "input" in tool_call
        assert "name" in tool_call

        anthropic_blocks.append(
            ToolUseBlockParam(
                id=tool_call["id"],
                input=tool_call["input"],
                name=tool_call["name"],
                type="tool_use",
            )
        )

    return anthropic_blocks


def messages_to_anthropic_messages(
    messages: Sequence[ChatMessage],
    cache_idx: Optional[int] = None,
    model: Optional[str] = None,
) -> Tuple[Sequence[MessageParam], str]:
    """
    Converts a list of generic ChatMessages to anthropic messages.

    Args:
        messages: List of ChatMessages
        cache_idx: Optional index to enable caching up to
        model: Optional model name used to validate prompt caching support

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
            if model is None or is_anthropic_prompt_caching_supported_model(model):
                message.additional_kwargs["cache_control"] = {"type": "ephemeral"}

        if message.role == MessageRole.SYSTEM:
            system_prompt.extend(
                blocks_to_anthropic_blocks(message.blocks, message.additional_kwargs)
            )
        elif message.role == MessageRole.FUNCTION or message.role == MessageRole.TOOL:
            anthropic_blocks = blocks_to_anthropic_blocks(
                message.blocks, message.additional_kwargs
            )
            content = ToolResultBlockParam(
                tool_use_id=message.additional_kwargs["tool_call_id"],
                type="tool_result",
                content=anthropic_blocks,
            )
            anth_message = MessageParam(
                role=MessageRole.USER.value,
                content=[content],
            )
            anthropic_messages.append(anth_message)
        else:
            content = blocks_to_anthropic_blocks(
                message.blocks, message.additional_kwargs
            )
            anth_message = MessageParam(
                role=message.role.value,
                content=content,
            )
            anthropic_messages.append(anth_message)

    return __merge_common_role_msgs(anthropic_messages), system_prompt


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


# Anthropic models that support prompt caching
# Based on: https://docs.claude.com/en/docs/build-with-claude/prompt-caching
ANTHROPIC_PROMPT_CACHING_SUPPORTED_MODELS: Tuple[str, ...] = (
    # Claude 4.1 Opus
    "claude-opus-4-1-20250805",
    "claude-opus-4-1",
    # Claude 4 Opus
    "claude-opus-4-20250514",
    "claude-opus-4-0",
    "claude-4-opus-20250514",
    # Claude 4.5 Sonnet
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5",
    # Claude 4 Sonnet
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-0",
    "claude-4-sonnet-20250514",
    # Claude 3.7 Sonnet
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-latest",
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-latest",
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    # Claude 3 Haiku
    "claude-3-haiku-20240307",
    "claude-3-haiku-latest",
    # Claude 3 Opus (deprecated)
    "claude-3-opus-20240229",
    "claude-3-opus-latest",
)


def is_anthropic_prompt_caching_supported_model(model: str) -> bool:
    """
    Check if the given Anthropic model supports prompt caching.

    Args:
        model: The Anthropic model identifier (e.g., 'claude-sonnet-4-20250514')

    Returns:
        True if the model supports prompt caching, False otherwise.

    See: https://docs.claude.com/en/docs/build-with-claude/prompt-caching

    """
    return model in ANTHROPIC_PROMPT_CACHING_SUPPORTED_MODELS
