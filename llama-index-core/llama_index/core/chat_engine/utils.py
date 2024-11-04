from typing import List
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    MessageRole,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.callbacks import CallbackManager
from llama_index.core.types import TokenGen, TokenAsyncGen


def get_prefix_messages_with_context(
    context_template: str,
    system_prompt: str,
    prefix_messages: List[ChatMessage],
    chat_history: List[ChatMessage],
    llm_metadata_system_role: MessageRole,
) -> List[ChatMessage]:
    context_str_w_sys_prompt = context_template + system_prompt.strip()
    return [
        ChatMessage(content=context_str_w_sys_prompt, role=llm_metadata_system_role),
        *prefix_messages,
        *chat_history,
        ChatMessage(content="{query_str}", role=MessageRole.USER),
    ]


def get_response_synthesizer(
    llm: LLM,
    callback_manager: CallbackManager,
    qa_messages: List[ChatMessage],
    refine_messages: List[ChatMessage],
    streaming: bool = False,
) -> CompactAndRefine:
    return CompactAndRefine(
        llm=llm,
        callback_manager=callback_manager,
        text_qa_template=ChatPromptTemplate.from_messages(qa_messages),
        refine_template=ChatPromptTemplate.from_messages(refine_messages),
        streaming=streaming,
    )


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
