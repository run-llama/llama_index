from typing import List, Sequence

from llama_index.llms.base import ChatMessage, MessageRole

HUMAN_PREFIX = "\n\nHuman:"
ASSISTANT_PREFIX = "\n\nAssistant:"


def _message_to_anthropic_prompt(message: ChatMessage) -> str:
    if message.role == MessageRole.USER:
        prompt = f"{HUMAN_PREFIX} {message.content}"
    elif message.role == MessageRole.ASSISTANT:
        prompt = f"{ASSISTANT_PREFIX} {message.content}"
    elif message.role == MessageRole.SYSTEM:
        prompt = f"{HUMAN_PREFIX} <system>{message.content}</system>"
    elif message.role == MessageRole.FUNCTION:
        raise ValueError(f"Message role {message.role} is not supported.")

    return prompt


def messages_to_anthropic_prompt(messages: Sequence[ChatMessage]) -> str:
    str_list = [_message_to_anthropic_prompt(message) for message in messages]
    prompt_str = "".join(str_list)

    # NOTE: make sure the prompt ends with the assistant prefix
    if not prompt_str.endswith(ASSISTANT_PREFIX):
        prompt_str += ASSISTANT_PREFIX
    return prompt_str
