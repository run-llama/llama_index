from typing import List

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.types import TokenGen


def response_gen_with_chat_history(
    message: str, chat_history: List[ChatMessage], response_gen: TokenGen
) -> TokenGen:
    response_str = ""
    for token in response_gen:
        response_str += token
        yield token

    chat_history.extend(
        [
            ChatMessage(role=MessageRole.USER, content=message),
            ChatMessage(role=MessageRole.ASSISTANT, content=response_str),
        ]
    )
