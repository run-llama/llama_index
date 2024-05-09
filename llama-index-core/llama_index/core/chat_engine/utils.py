from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    MessageRole,
)
from llama_index.core.types import TokenGen, TokenAsyncGen


def response_gen_from_query_engine(response_gen: TokenGen) -> ChatResponseGen:
    response_str = ""
    for token in response_gen:
        response_str += token
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response_str),
            delta=token,
        )


async def aresponse_gen_from_query_engine(
    response_gen: TokenAsyncGen,
) -> ChatResponseAsyncGen:
    response_str = ""
    async for token in response_gen:
        response_str += token
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response_str),
            delta=token,
        )
