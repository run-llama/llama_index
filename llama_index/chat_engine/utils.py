from typing import Optional
from llama_index.chat_engine.types import ChatHistoryType
from langchain.memory import ChatMessageHistory
from langchain.chat_models.base import BaseChatModel

from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor


def to_chat_buffer(chat_history: ChatHistoryType) -> str:
    buffer = ""
    for human_s, ai_s in chat_history:
        human = "Human: " + human_s
        ai = "Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer


def to_langchain_chat_history(
    chat_history: Optional[ChatHistoryType] = None,
) -> ChatMessageHistory:
    history = ChatMessageHistory()
    if chat_history is not None:
        for human_message, ai_message in chat_history:
            history.add_user_message(human_message)
            history.add_ai_message(ai_message)
    return history


def is_chat_model(service_context: ServiceContext) -> bool:
    llm_predictor = service_context.llm_predictor
    if not isinstance(llm_predictor, LLMPredictor):
        return False
    try:
        return isinstance(llm_predictor.llm, BaseChatModel)
    except AttributeError:
        # NOTE: in testing, our mock llm predictor doesn't have llm attribute
        return False
