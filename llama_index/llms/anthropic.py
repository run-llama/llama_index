from typing import Any, Dict, Optional, Sequence

from llama_index.llms.anthropic_utils import (
    anthropic_modelname_to_contextsize,
    messages_to_anthropic_prompt,
)
from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.llms.generic_utils import (
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)


class Anthropic(LLM):
    def __init__(
        self,
        model: str = "claude-2",
        temperature: float = 0.0,
        max_tokens: int = 256,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 10,
        additional_kwargs: Dict[str, Any] = {},
    ) -> None:
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError:
            raise ImportError(
                "You must install the `anthropic` package to use Anthropic."
            )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._additional_kwargs = additional_kwargs

        self._client = Anthropic(
            base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        self._aclient = AsyncAnthropic(
            base_url=base_url, timeout=timeout, max_retries=max_retries
        )

    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=anthropic_modelname_to_contextsize(self._model),
            num_output=self._max_tokens,
            is_chat_model=True,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens_to_sample": self._max_tokens,
        }
        model_kwargs = {
            **base_kwargs,
            **self._additional_kwargs,
        }
        return model_kwargs

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = messages_to_anthropic_prompt(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.completions.create(
            prompt=prompt, stream=False, **all_kwargs
        )
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content=response.completion
            ),
            raw=dict(response),
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = messages_to_anthropic_prompt(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._client.completions.create(
            prompt=prompt, stream=True, **all_kwargs
        )

        def gen() -> ChatResponseGen:
            content = ""
            role = MessageRole.ASSISTANT
            for r in response:
                content_delta = r.completion
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=r,
                )

        return gen()

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError()

    def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError()

    def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError()

    def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError()
