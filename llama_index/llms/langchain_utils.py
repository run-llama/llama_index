

from typing import Sequence
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.llms import Cohere, AI21, OpenAI

from llama_index.constants import AI21_J2_CONTEXT_WINDOW, COHERE_CONTEXT_WINDOW
from llama_index.llms.base import LLMMetadata, Message
from llama_index.llms.openai_utils import openai_modelname_to_contextsize
from langchain.schema import BaseMessage as LCMessage


def is_chat_model(llm: BaseLanguageModel):
    return isinstance(llm, BaseChatModel)


def to_lc_messages(messages: Sequence[Message]) -> Sequence[LCMessage]:
    return []


def from_lc_messages(messages: Sequence[LCMessage]) -> Sequence[Message]:
    return []

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
