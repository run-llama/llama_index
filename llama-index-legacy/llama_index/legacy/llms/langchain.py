from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Sequence

if TYPE_CHECKING:
    from langchain.base_language import BaseLanguageModel

from llama_index.legacy.bridge.pydantic import PrivateAttr
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.legacy.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.legacy.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.legacy.llms.llm import LLM
from llama_index.legacy.types import BaseOutputParser, PydanticProgramMode


class LangChainLLM(LLM):
    """Adapter for a LangChain LLM."""

    _llm: Any = PrivateAttr()

    def __init__(
        self,
        llm: "BaseLanguageModel",
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        self._llm = llm
        super().__init__(
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "LangChainLLM"

    @property
    def llm(self) -> "BaseLanguageModel":
        return self._llm

    @property
    def metadata(self) -> LLMMetadata:
        from llama_index.legacy.llms.langchain_utils import get_llm_metadata

        return get_llm_metadata(self._llm)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        from llama_index.legacy.llms.langchain_utils import (
            from_lc_messages,
            to_lc_messages,
        )

        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            completion_response = self.complete(prompt, formatted=True, **kwargs)
            return completion_response_to_chat_response(completion_response)

        lc_messages = to_lc_messages(messages)
        lc_message = self._llm.predict_messages(messages=lc_messages, **kwargs)
        message = from_lc_messages([lc_message])[0]
        return ChatResponse(message=message)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        output_str = self._llm.predict(prompt, **kwargs)
        return CompletionResponse(text=output_str)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        if not self.metadata.is_chat_model:
            prompt = self.messages_to_prompt(messages)
            stream_completion = self.stream_complete(prompt, formatted=True, **kwargs)
            return stream_completion_response_to_chat_response(stream_completion)

        if hasattr(self._llm, "stream"):

            def gen() -> Generator[ChatResponse, None, None]:
                from llama_index.legacy.llms.langchain_utils import (
                    from_lc_messages,
                    to_lc_messages,
                )

                lc_messages = to_lc_messages(messages)
                response_str = ""
                for message in self._llm.stream(lc_messages, **kwargs):
                    message = from_lc_messages([message])[0]
                    delta = message.content
                    response_str += delta
                    yield ChatResponse(
                        message=ChatMessage(role=message.role, content=response_str),
                        delta=delta,
                    )

            return gen()

        else:
            from llama_index.legacy.langchain_helpers.streaming import (
                StreamingGeneratorCallbackHandler,
            )

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

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        from llama_index.legacy.langchain_helpers.streaming import (
            StreamingGeneratorCallbackHandler,
        )

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

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        # TODO: Implement async chat
        return self.chat(messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        # TODO: Implement async complete
        return self.complete(prompt, formatted=formatted, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        # TODO: Implement async stream_chat

        async def gen() -> ChatResponseAsyncGen:
            for message in self.stream_chat(messages, **kwargs):
                yield message

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        # TODO: Implement async stream_complete

        async def gen() -> CompletionResponseAsyncGen:
            for response in self.stream_complete(prompt, formatted=formatted, **kwargs):
                yield response

        return gen()
