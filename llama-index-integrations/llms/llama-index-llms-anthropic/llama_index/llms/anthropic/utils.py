from typing import Dict, Sequence, Tuple

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from anthropic.types import MessageParam, TextBlockParam, ImageBlockParam
from anthropic.types.tool_result_block_param import ToolResultBlockParam
from anthropic.types.tool_use_block_param import ToolUseBlockParam
from anthropic.types.beta.prompt_caching import (
    PromptCachingBetaTextBlockParam,
    PromptCachingBetaCacheControlEphemeralParam,
)

HUMAN_PREFIX = "\n\nHuman:"
ASSISTANT_PREFIX = "\n\nAssistant:"

CLAUDE_MODELS: Dict[str, int] = {
    "claude-instant-1": 100000,
    "claude-instant-1.2": 100000,
    "claude-2": 100000,
    "claude-2.0": 100000,
    "claude-2.1": 200000,
    "claude-3-opus-latest": 180000,
    "claude-3-opus-20240229": 180000,
    "claude-3-opus@20240229": 180000,  # Alternate name for Vertex AI
    "anthropic.claude-3-opus-20240229-v1:0": 180000,  # Alternate name for Bedrock
    "claude-3-sonnet-latest": 180000,
    "claude-3-sonnet-20240229": 180000,
    "claude-3-sonnet@20240229": 180000,  # Alternate name for Vertex AI
    "anthropic.claude-3-sonnet-20240229-v1:0": 180000,  # Alternate name for Bedrock
    "claude-3-haiku-latest": 180000,
    "claude-3-haiku-20240307": 180000,
    "claude-3-haiku@20240307": 180000,  # Alternate name for Vertex AI
    "anthropic.claude-3-haiku-20240307-v1:0": 180000,  # Alternate name for Bedrock
    "claude-3-5-sonnet-latest": 180000,
    "claude-3-5-sonnet-20240620": 180000,
    "claude-3-5-sonnet-20241022": 180000,
    "claude-3-5-sonnet-v2@20241022": 180000,  # Alternate name for Vertex AI
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 180000,  # Alternate name for Bedrock
    "claude-3-5-sonnet@20240620": 180000,  # Alternate name for Vertex AI
    "claude-3-5-haiku-20241022": 180000,
    "claude-3-5-haiku@20241022": 180000,  # Alternate name for Vertex AI
    "anthropic.claude-3-5-haiku-20241022-v1:0": 180000,  # Alternate name for Bedrock
}


def is_function_calling_model(modelname: str) -> bool:
    return "claude-3" in modelname


def anthropic_modelname_to_contextsize(modelname: str) -> int:
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
) -> Tuple[Sequence[MessageParam], str]:
    """Converts a list of generic ChatMessages to anthropic messages.

    Args:
        messages: List of ChatMessages

    Returns:
        Tuple of:
        - List of anthropic messages
        - System prompt
    """
    anthropic_messages = []
    system_prompt = ""
    for message in messages:
        if message.role == MessageRole.SYSTEM:
            system_prompt += message.content + "\n"
        elif message.role == MessageRole.FUNCTION or message.role == MessageRole.TOOL:
            content = ToolResultBlockParam(
                tool_use_id=message.additional_kwargs["tool_call_id"],
                type="tool_result",
                content=[TextBlockParam(text=message.content, type="text")],
            )
            anth_message = MessageParam(
                role=MessageRole.USER.value,
                content=[content],
            )
            anthropic_messages.append(anth_message)
        else:
            content = []
            if message.content and isinstance(message.content, list):
                for item in message.content:
                    if item and isinstance(item, dict) and item.get("type", None):
                        if item["type"] == "image":
                            content.append(ImageBlockParam(**item))
                        elif "cache_control" in item and item["type"] == "text":
                            content.append(
                                PromptCachingBetaTextBlockParam(
                                    text=item["text"],
                                    type="text",
                                    cache_control=PromptCachingBetaCacheControlEphemeralParam(
                                        type="ephemeral"
                                    ),
                                )
                            )
                        else:
                            content.append(TextBlockParam(**item))
                    else:
                        content.append(TextBlockParam(text=item, type="text"))
            elif message.content:
                content_ = (
                    PromptCachingBetaTextBlockParam(
                        text=message.content,
                        type="text",
                        cache_control=PromptCachingBetaCacheControlEphemeralParam(
                            type="ephemeral"
                        ),
                    )
                    if "cache_control" in message.additional_kwargs
                    else TextBlockParam(text=message.content, type="text")
                )
                content.append(content_)

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
                content=content,  # TODO: type detect for multimodal
            )
            anthropic_messages.append(anth_message)
    return __merge_common_role_msgs(anthropic_messages), system_prompt.strip()


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
