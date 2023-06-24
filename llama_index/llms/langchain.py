from typing import Any, Coroutine, Sequence
from pydantic import BaseModel
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

        raise NotImplementedError()

        # handler = StreamingGeneratorCallbackHandler()

        # if not hasattr(self._llm, "callbacks"):
        #     raise ValueError("LLM must support callbacks to use streaming.")

        # self._llm.callbacks = [handler]

        # if not getattr(self._llm, "streaming", False):
        #     raise ValueError("LLM must support streaming and set streaming=True.")

        # thread = Thread(target=self._predict, args=[prompt], kwargs=prompt_args)
        # thread.start()

        # response_gen = handler.get_response_gen()

    async def achat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> Coroutine[Any, Any, ChatResponseType]:
        return await super().achat(messages, **kwargs)

    async def acomplete(
        self, prompt: str, **kwargs: Any
    ) -> Coroutine[Any, Any, CompletionResponseType]:
        return await super().acomplete(prompt, **kwargs)
