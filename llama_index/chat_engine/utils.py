from llama_index.llms.base import (
    ChatMessage,
    MessageRole,
    ChatResponse,
    ChatResponseGen,
)
from llama_index.memory import BaseMemory
from llama_index.types import TokenGen


def response_gen_with_chat_history(
    message: str, memory: BaseMemory, response_gen: TokenGen
) -> ChatResponseGen:
    response_str = ""
    for token in response_gen:
        response_str += token
        yield ChatResponse(
            role=MessageRole.ASSISTANT, content=response_str, delta=token
        )

    # Record response
    memory.put(ChatMessage(role=MessageRole.USER, content=message))
    memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_str))
