from typing import Sequence

from llama_index.llms.base import ChatMessage, MessageRole

HUMAN_PREFIX = "\n\nHuman:"
ASSISTANT_PREFIX = "\n\nAssistant:"


CLAUDE_MODELS = {
    "claude-instant-1": 100000,
    "claude-instant-1.2": 100000,
    "claude-2": 100000,
    "claude-2.0": 100000,
}


def anthropic_modelname_to_contextsize(modelname: str) -> int:
    if modelname not in CLAUDE_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Anthropic model name."
            "Known models are: " + ", ".join(CLAUDE_MODELS.keys())
        )

    return CLAUDE_MODELS[modelname]


def _message_to_anthropic_prompt(message: ChatMessage) -> str:
    if message.role == MessageRole.USER:
        prompt = f"{HUMAN_PREFIX} {message.content}"
    elif message.role == MessageRole.ASSISTANT:
        prompt = f"{ASSISTANT_PREFIX} {message.content}"
    elif message.role == MessageRole.SYSTEM:
        prompt = f"{HUMAN_PREFIX} <system>{message.content}</system>"
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
