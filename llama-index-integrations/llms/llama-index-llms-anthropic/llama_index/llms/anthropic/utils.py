from typing import Dict, Sequence, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from anthropic.types import MessageParam, TextBlockParam

HUMAN_PREFIX = "\n\nHuman:"
ASSISTANT_PREFIX = "\n\nAssistant:"

CLAUDE_MODELS: Dict[str, int] = {
    "claude-instant-1": 100000,
    "claude-instant-1.2": 100000,
    "claude-2": 100000,
    "claude-2.0": 100000,
    "claude-2.1": 200000,
    "claude-3-opus-20240229": 180000,
    "claude-3-sonnet-20240229": 180000,
    "claude-3-haiku-20240307": 180000,
}


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
            system_prompt = message.content
        else:
            message = MessageParam(
                role=message.role.value,
                content=[
                    TextBlockParam(text=message.content, type="text")
                ],  # TODO: type detect for multimodal
            )
            anthropic_messages.append(message)
    return __merge_common_role_msgs(anthropic_messages), system_prompt


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
