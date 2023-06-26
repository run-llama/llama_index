from typing import List, Sequence

from llama_index.bridge.langchain import (
    BaseLanguageModel,
    ChatOpenAI,
    BaseChatModel,
    Cohere,
    AI21,
    OpenAI,
    BaseMessage as LCMessage,
    HumanMessage,
    AIMessage,
    FunctionMessage as LCFunctionMessage,
)

from llama_index.constants import AI21_J2_CONTEXT_WINDOW, COHERE_CONTEXT_WINDOW
from llama_index.llms.base import ChatMessage, LLMMetadata, Message, FunctionMessage
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
            assert isinstance(message, FunctionMessage)
            lc_messages.append(
                LCFunctionMessage(
                    content=message.content,
                    additional_kwargs=message.additional_kwargs,
                    name=message.name,
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
                    role="user",
                )
            )
        elif isinstance(lc_message, AIMessage):
            messages.append(
                ChatMessage(
                    content=lc_message.content,
                    additional_kwargs=lc_message.additional_kwargs,
                    role="assistant",
                )
            )
        elif isinstance(lc_message, LCFunctionMessage):
            messages.append(
                FunctionMessage(
                    content=lc_message.content,
                    additional_kwargs=lc_message.additional_kwargs,
                    name=lc_message.name,
                    role="function",
                )
            )
        else:
            raise ValueError(f"Invalid message type: {type(lc_message)}")
    return messages


def get_llm_metadata(llm: BaseLanguageModel) -> LLMMetadata:
    """Get LLM metadata from llm."""
    if not isinstance(llm, BaseLanguageModel):
        raise ValueError("llm must be an instance of langchain.llms.base.LLM")
    if isinstance(llm, OpenAI):
        return LLMMetadata(
            context_window=openai_modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens,
        )
    elif isinstance(llm, ChatOpenAI):
        return LLMMetadata(
            context_window=openai_modelname_to_contextsize(llm.model_name),
            num_output=llm.max_tokens or -1,
        )
    elif isinstance(llm, Cohere):
        # June 2023: Cohere's supported max input size for Generation models is 2048
        # Reference: <https://docs.cohere.com/docs/tokens>
        return LLMMetadata(
            context_window=COHERE_CONTEXT_WINDOW, num_output=llm.max_tokens
        )
    elif isinstance(llm, AI21):
        # June 2023:
        #   AI21's supported max input size for
        #   J2 models is 8K (8192 tokens to be exact)
        # Reference: <https://docs.ai21.com/changelog/increased-context-length-for-j2-foundation-models>  # noqa
        return LLMMetadata(
            context_window=AI21_J2_CONTEXT_WINDOW, num_output=llm.maxTokens
        )
    else:
        return LLMMetadata()
