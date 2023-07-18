from llama_index.llms.base import (
    ChatMessage,
    MessageRole,
    ChatResponse,
    ChatResponseGen,
)
from llama_index.types import TokenGen


def response_gen_from_query_engine(response_gen: TokenGen) -> ChatResponseGen:
    def gen() -> ChatResponseGen:
        response_str = ""
        for token in response_gen:
            response_str += token
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=response_str),
                delta=token,
            )

    return gen()
