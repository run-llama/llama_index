from typing import Dict, Sequence, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole

CLAUDE_MODELS: Dict[str, int] = {
    "claude-instant-1": 100000,
    "claude-instant-1.2": 100000,
    "claude-2": 100000,
    "claude-2.0": 100000,
    "claude-2.1": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
}


def anthropic_modelname_to_contextsize(modelname: str) -> int:
    if modelname not in CLAUDE_MODELS:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Anthropic model name."
            "Known models are: " + ", ".join(CLAUDE_MODELS.keys())
        )

    return CLAUDE_MODELS[modelname]


def messages_to_anthropic_messages(
    messages: Sequence[ChatMessage],
) -> Tuple[Sequence[ChatMessage], str]:
    anthropic_messages = []
    system_prompt = ""
    for message in messages:
        if message.role == MessageRole.SYSTEM:
            system_prompt = message.content
        else:
            anthropic_messages.append(message)
    return anthropic_messages, system_prompt
