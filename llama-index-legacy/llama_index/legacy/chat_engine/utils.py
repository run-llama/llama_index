from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    MessageRole,
)
from llama_index.types import TokenGen


def response_gen_from_query_engine(response_gen: TokenGen) -> ChatResponseGen:
    response_str = ""
    for token in response_gen:
        response_str += token
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response_str),
            delta=token,
        )
