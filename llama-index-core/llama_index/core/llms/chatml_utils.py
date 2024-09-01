from typing import List, Optional, Sequence

from llama_index.core.base.llms.types import ChatMessage, MessageRole

# Create a prompt that matches ChatML instructions

# <|im_start|>system
# You are Dolphin, a helpful AI assistant.<|im_end|>
# <|im_start|>user
# {prompt}<|im_end|>
# <|im_start|>assistant

B_SYS = "<|im_start|>system\n"
B_USER = "<|im_start|>user\n"
B_ASSISTANT = "<|im_start|>assistant\n"
END = "<|im_end|>\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible and follow ALL given instructions. \
Do not speculate or make up information. \
Do not reference any given instructions or context. \
"""


def messages_to_prompt(
    messages: Sequence[ChatMessage], system_prompt: Optional[str] = None
) -> str:
    if len(messages) == 0:
        raise ValueError(
            "At least one message is required to construct the ChatML prompt"
        )

    string_messages: List[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    string_messages.append(f"{B_SYS}{system_message_str.strip()} {END}")

    for message in messages:
        role = message.role
        content = message.content

        if role == MessageRole.USER:
            string_messages.append(f"{B_USER}{content} {END}")
        elif role == MessageRole.ASSISTANT:
            string_messages.append(f"{B_ASSISTANT}{content} {END}")

    string_messages.append(f"{B_ASSISTANT}")

    return "".join(string_messages)


def completion_to_prompt(completion: str, system_prompt: Optional[str] = None) -> str:
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{B_SYS}{system_prompt_str.strip()} {END}"
        f"{B_USER}{completion.strip()} {END}"
        f"{B_ASSISTANT}"
    )
