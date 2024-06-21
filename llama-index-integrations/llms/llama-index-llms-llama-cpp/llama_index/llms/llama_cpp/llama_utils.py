from typing import List, Optional, Sequence

from llama_index.core.base.llms.types import ChatMessage, MessageRole

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
    string_messages: List[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    system_message_str = f"{B_SYS} {system_message_str.strip()} {E_SYS}"

    for i in range(0, len(messages), 2):
        # first message should always be a user
        user_message = messages[i]
        assert user_message.role == MessageRole.USER

        if i == 0:
            # make sure system prompt is included at the start
            str_message = f"{BOS} {B_INST} {system_message_str} "
        else:
            # end previous user-assistant interaction
            string_messages[-1] += f" {EOS}"
            # no need to include system prompt
            str_message = f"{BOS} {B_INST} "

        # include user message content
        str_message += f"{user_message.content} {E_INST}"

        if len(messages) > (i + 1):
            # if assistant message exists, add to str_message
            assistant_message = messages[i + 1]
            assert assistant_message.role == MessageRole.ASSISTANT
            str_message += f" {assistant_message.content}"

        string_messages.append(str_message)

    return "".join(string_messages)


def completion_to_prompt(completion: str, system_prompt: Optional[str] = None) -> str:
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{BOS} {B_INST} {B_SYS} {system_prompt_str.strip()} {E_SYS} "
        f"{completion.strip()} {E_INST}"
    )


HEADER_SYS = "<|start_header_id|>system<|end_header_id|>\n\n"
HEADER_USER = "<|start_header_id|>user<|end_header_id|>\n\n"
HEADER_ASSIST = "<|start_header_id|>assistant<|end_header_id|>\n\n"
EOT = "<|eot_id|>\n"


def messages_to_prompt_v3_instruct(
    messages: Sequence[ChatMessage], system_prompt: Optional[str] = None
) -> str:
    """
    Convert a sequence of chat messages to Llama 3 Instruct format.

    Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

    Note: `<|begin_of_text|>` is not needed as Llama.cpp appears to add it already.
    """
    string_messages: List[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    # make sure system prompt is included at the start
    string_messages.append(f"{HEADER_SYS}{system_message_str.strip()}{EOT}")

    for i in range(0, len(messages), 2):
        # first message should always be a user
        user_message = messages[i]
        assert user_message.role == MessageRole.USER
        # include user message content
        str_message = f"{HEADER_USER}{user_message.content}{EOT}"

        if len(messages) > (i + 1):
            # if assistant message exists, add to str_message
            assistant_message = messages[i + 1]
            assert assistant_message.role == MessageRole.ASSISTANT
            str_message += f"{HEADER_ASSIST}{assistant_message.content}{EOT}"

        string_messages.append(str_message)

    # prompt the LLM to begin its response
    string_messages.append(HEADER_ASSIST)

    return "".join(string_messages)


def completion_to_prompt_v3_instruct(
    completion: str, system_prompt: Optional[str] = None
) -> str:
    """
    Convert completion instruction string to Llama 3 Instruct format.

    Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

    Note: `<|begin_of_text|>` is not needed as Llama.cpp appears to add it already.
    """
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{HEADER_SYS}{system_prompt_str.strip()}{EOT}"
        f"{HEADER_USER}{completion.strip()}{EOT}"
        f"{HEADER_ASSIST}"
    )
