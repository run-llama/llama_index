from threading import Thread
from typing import Any, Generator, Sequence

from langchain.base_language import BaseLanguageModel

from llama_index.langchain_helpers.streaming import StreamingGeneratorCallbackHandler
from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.langchain_utils import (
    from_lc_messages,
    get_llm_metadata,
    to_lc_messages,
)


class LangChainLLM(LLM):
    """Adapter for a LangChain LLM."""

    def __init__(self, llm: BaseLanguageModel) -> None:
        self._llm = llm

    @property
    def llm(self) -> BaseLanguageModel:
        return self._llm

    @property
    def metadata(self) -> LLMMetadata:
        return get_llm_metadata(self._llm)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        lc_messages = to_lc_messages(messages)
        lc_message = self._llm.predict_messages(messages=lc_messages, **kwargs)
        message = from_lc_messages([lc_message])[0]
        return ChatResponse(message=message)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        output_str = self._llm.predict(prompt, **kwargs)
        return CompletionResponse(text=output_str)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        handler = StreamingGeneratorCallbackHandler()

        if not hasattr(self._llm, "streaming"):
            raise ValueError("LLM must support streaming.")
        if not hasattr(self._llm, "callbacks"):
            raise ValueError("LLM must support callbacks to use streaming.")

        self._llm.callbacks = [handler]  # type: ignore
        self._llm.streaming = True  # type: ignore

        thread = Thread(target=self.chat, args=[messages], kwargs=kwargs)
        thread.start()

        response_gen = handler.get_response_gen()

        def gen() -> Generator[ChatResponse, None, None]:
            text = ""
            for delta in response_gen:
                text += delta
                yield ChatResponse(
                    message=ChatMessage(text=text),
                    delta=delta,
                )

        return gen()

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        handler = StreamingGeneratorCallbackHandler()

        if not hasattr(self._llm, "streaming"):
            raise ValueError("LLM must support streaming.")
        if not hasattr(self._llm, "callbacks"):
            raise ValueError("LLM must support callbacks to use streaming.")

        self._llm.callbacks = [handler]  # type: ignore
        self._llm.streaming = True  # type: ignore

        thread = Thread(target=self.complete, args=[prompt], kwargs=kwargs)
        thread.start()

        response_gen = handler.get_response_gen()

        def gen() -> Generator[CompletionResponse, None, None]:
            text = ""
            for delta in response_gen:
                text += delta
                yield CompletionResponse(delta=delta, text=text)

        return gen()

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        # TODO: Implement async chat
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # TODO: Implement async complete
        return self.complete(prompt, **kwargs)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        # TODO: Implement async stream_chat
        return self.stream_chat(messages, **kwargs)

    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        # TODO: Implement async stream_complete
        return self.stream_complete(prompt, **kwargs)
