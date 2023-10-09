from typing import List, Sequence

from llama_index.bridge.langchain import (
    AI21,
    AIMessage,
    BaseChatModel,
    BaseLanguageModel,
    ChatAnyscale,
    ChatOpenAI,
    Cohere,
    FunctionMessage,
    HumanMessage,
    OpenAI,
    SystemMessage,
)
from llama_index.bridge.langchain import BaseMessage as LCMessage
from llama_index.constants import AI21_J2_CONTEXT_WINDOW, COHERE_CONTEXT_WINDOW
from llama_index.llms.anyscale_utils import anyscale_modelname_to_contextsize
from llama_index.llms.base import ChatMessage, LLMMetadata, MessageRole
from llama_index.llms.openai_utils import openai_modelname_to_contextsize


def is_chat_model(llm: BaseLanguageModel) -> bool:
    return isinstance(llm, BaseChatModel)


def to_lc_messages(messages: Sequence[ChatMessage]) -> List[LCMessage]:
    lc_messages: List[LCMessage] = []
    for message in messages:
        if message.role == "user":
            lc_messages.append(
                HumanMessage(
                    content=message.content, additional_kwargs=message.additional_kwargs
                )
            )
        elif message.role == "assistant":
            lc_messages.append(
                AIMessage(
                    content=message.content, additional_kwargs=message.additional_kwargs
                )
            )
        elif message.role == "function":
            if "name" not in message.additional_kwargs:
                raise ValueError("name cannot be None for function message.")
            name = message.additional_kwargs.pop("name")
            lc_messages.append(
                FunctionMessage(
                    content=message.content,
                    additional_kwargs=message.additional_kwargs,
                    name=name,
                )
            )
        elif message.role == "system":
            lc_messages.append(
                SystemMessage(
                    content=message.content, additional_kwargs=message.additional_kwargs
                )
            )
        else:
            raise ValueError(f"Invalid role: {message.role}")
    return lc_messages


def from_lc_messages(lc_messages: Sequence[LCMessage]) -> List[ChatMessage]:
    messages: List[ChatMessage] = []
    for lc_message in lc_messages:
        if isinstance(lc_message, HumanMessage):
            messages.append(
                ChatMessage(
                    content=lc_message.content,
                    additional_kwargs=lc_message.additional_kwargs,
                    role=MessageRole.USER,
                )
            )
        elif isinstance(lc_message, AIMessage):
            messages.append(
                ChatMessage(
                    content=lc_message.content,
                    additional_kwargs=lc_message.additional_kwargs,
                    role=MessageRole.ASSISTANT,
                )
            )
        elif isinstance(lc_message, FunctionMessage):
            messages.append(
                ChatMessage(
                    content=lc_message.content,
                    additional_kwargs=lc_message.additional_kwargs,
                    name=lc_message.name,
                    role=MessageRole.FUNCTION,
                )
            )
        elif isinstance(lc_message, SystemMessage):
            messages.append(
                ChatMessage(
                    content=lc_message.content,
                    additional_kwargs=lc_message.additional_kwargs,
                    role=MessageRole.SYSTEM,
                )
            )
        else:
            raise ValueError(f"Invalid message type: {type(lc_message)}")
    return messages


def get_llm_metadata(llm: BaseLanguageModel) -> LLMMetadata:
    """Get LLM metadata from llm."""
    if not isinstance(llm, BaseLanguageModel):
        raise ValueError("llm must be an instance of langchain.llms.base.LLM")

    is_chat_model_ = is_chat_model(llm)

    if isinstance(llm, OpenAI):
        return LLMMetadata(
            context_window=openai_modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens,
            is_chat_model=is_chat_model_,
            model_name=llm.model_name,
        )
    elif isinstance(llm, ChatAnyscale):
        return LLMMetadata(
            context_window=anyscale_modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens or -1,
            is_chat_model=is_chat_model_,
            model_name=llm.model_name,
        )
    elif isinstance(llm, ChatOpenAI):
        return LLMMetadata(
            context_window=openai_modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens or -1,
            is_chat_model=is_chat_model_,
            model_name=llm.model_name,
        )
    elif isinstance(llm, Cohere):
        # June 2023: Cohere's supported max input size for Generation models is 2048
        # Reference: <https://docs.cohere.com/docs/tokens>
        return LLMMetadata(
            context_window=COHERE_CONTEXT_WINDOW,
            num_output=llm.max_tokens,
            is_chat_model=is_chat_model_,
            model_name=llm.model,
        )
    elif isinstance(llm, AI21):
        # June 2023:
        #   AI21's supported max input size for
        #   J2 models is 8K (8192 tokens to be exact)
        # Reference: <https://docs.ai21.com/changelog/increased-context-length-for-j2-foundation-models>
        return LLMMetadata(
            context_window=AI21_J2_CONTEXT_WINDOW,
            num_output=llm.maxTokens,
            is_chat_model=is_chat_model_,
            model_name=llm.model,
        )
    else:
        return LLMMetadata(is_chat_model=is_chat_model_)
