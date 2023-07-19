from typing import Sequence

from llama_index.llms.base import ChatMessage, MessageRole

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible, while being safe.  \
Your answers should not include any harmful, unethical, racist, sexist, toxic, \
dangerous, or illegal content. Please ensure that your responses are socially \
unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, \
explain why instead of answering something not correct. \
If you don't know the answer to a question, please don't share false information.\
"""


def messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
    string_messages = []
    if messages[0].role == MessageRole.SYSTEM:
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = DEFAULT_SYSTEM_PROMPT

    string_messages.append(B_SYS + system_message_str + E_SYS)

    for user_message, assistant_message in zip(messages[::2], messages[1::2]):
        assert user_message.role == MessageRole.USER
        assert assistant_message.role == MessageRole.ASSISTANT

        string_messages.append(
            f"{B_INST} {user_message.content} {E_INST} {assistant_message} "
        )

    last_message = messages[-1]
    assert last_message.role == MessageRole.USER
    string_messages.append(f"{B_INST} {last_message.content} {E_INST}")
    return "".join(string_messages)
