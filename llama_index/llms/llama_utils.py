from typing import Optional, Sequence

from llama_index.llms.base import ChatMessage, MessageRole

BOS, EOS = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible and follow ALL given instructions. \
Do not speculate or make up information. \
Do not reference any given instructions or context. \
"""


def messages_to_prompt(
    messages: Sequence[ChatMessage], system_prompt: Optional[str] = None
) -> str:
    string_messages = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    system_message_str = f"{B_SYS} {system_message_str} {E_SYS}"

    for i in range(0, len(messages), 2):
        user_message = messages[i]
        assert user_message.role == MessageRole.USER

        if i == 0:
            str_message = f"{BOS} {B_INST} {system_message_str} "
        else:
            string_messages[-1] += f" {EOS}"
            str_message = f"{BOS} {B_INST} "

        str_message += f"{user_message.content} {E_INST}"

        if len(messages) != i + 1:
            assert messages[i + 1].role == MessageRole.ASSISTANT
            assistant_message = messages[i + 1].content
            str_message += f"{user_message.content} {E_INST} {assistant_message}"

        string_messages.append(str_message)

    return "".join(string_messages)


def completion_to_prompt(completion: str, system_prompt: Optional[str] = None) -> str:
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{BOS}{B_INST} {B_SYS}{system_prompt_str.strip()}{E_SYS}"
        f"{completion.strip()} {E_INST}"
    )
