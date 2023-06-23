from typing import Any, Coroutine, Sequence
from pydantic import BaseModel, Model
from langchain.base_language import BaseLanguageModel

from llama_index.llms.base import (
    LLM,
    ChatResponseType,
    CompletionResponseType,
    LLMMetadata,
    Message,
)
from llama_index.llms.langchain_utils import get_llm_metadata


class LangChainLLM(LLM, BaseModel):
    llm: BaseLanguageModel

    def metadata(self) -> LLMMetadata:
        return get_llm_metadata(self.llm)

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponseType:
        return super().chat(messages, **kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponseType:
        return super().complete(prompt, **kwargs)

    async def achat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> Coroutine[Any, Any, ChatResponseType]:
        return await super().achat(messages, **kwargs)

    async def acomplete(
        self, prompt: str, **kwargs: Any
    ) -> Coroutine[Any, Any, CompletionResponseType]:
        return await super().acomplete(prompt, **kwargs)
