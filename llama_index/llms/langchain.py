from threading import Thread
from typing import Any, Generator, Sequence
from pydantic import BaseModel
from langchain.base_language import BaseLanguageModel
from llama_index.langchain_helpers.streaming import StreamingGeneratorCallbackHandler

from llama_index.llms.base import (
    LLM,
    ChatDeltaResponse,
    ChatMessage,
    ChatResponse,
    CompletionDeltaResponse,
    CompletionResponse,
    CompletionResponseType,
    LLMMetadata,
    Message,
    StreamChatResponse,
    StreamCompletionResponse,
)
from llama_index.llms.langchain_utils import (
    from_lc_messages,
    get_llm_metadata,
    to_lc_messages,
)


class LangChainLLM(LLM, BaseModel):
    llm: BaseLanguageModel

    @property
    def metadata(self) -> LLMMetadata:
        return get_llm_metadata(self.llm)

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        lc_messages = to_lc_messages(messages)
        lc_message = self.llm.predict_messages(messages=lc_messages, **kwargs)
        message = from_lc_messages([lc_message])[0]
        return ChatResponse(message=message)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        output_str = self.llm.predict(prompt, **kwargs)
        return CompletionResponseType(text=output_str)

    def stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> StreamChatResponse:
        handler = StreamingGeneratorCallbackHandler()

        if not hasattr(self.llm, "streaming"):
            raise ValueError("LLM must support streaming.")
        if not hasattr(self.llm, "callbacks"):
            raise ValueError("LLM must support callbacks to use streaming.")

        self.llm.callbacks = [handler]
        self.llm.streaming = True

        thread = Thread(target=self.chat, args=[messages], kwargs=kwargs)
        thread.start()

        response_gen = handler.get_response_gen()

        def gen() -> Generator[ChatDeltaResponse, None, None]:
            text = ""
            for delta in response_gen:
                text += delta
                yield ChatDeltaResponse(
                    message=ChatMessage(text=text),
                    delta=delta,
                )

        return gen()

    def stream_complete(self, prompt: str, **kwargs: Any) -> StreamCompletionResponse:
        handler = StreamingGeneratorCallbackHandler()

        if not hasattr(self.llm, "streaming"):
            raise ValueError("LLM must support streaming.")
        if not hasattr(self.llm, "callbacks"):
            raise ValueError("LLM must support callbacks to use streaming.")

        self.llm.callbacks = [handler]
        self.llm.streaming = True

        thread = Thread(target=self.complete, args=[prompt], kwargs=kwargs)
        thread.start()

        response_gen = handler.get_response_gen()

        def gen() -> Generator[CompletionDeltaResponse, None, None]:
            text = ""
            for delta in response_gen:
                text += delta
                yield CompletionDeltaResponse(delta=delta, text=text)

        return gen()

    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    async def astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> StreamChatResponse:
        return self.stream_chat(messages, **kwargs)

    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> StreamCompletionResponse:
        return self.stream_complete(prompt, **kwargs)
