from typing import Any, Dict, Generator, Optional, Sequence

from pydantic import BaseModel, Field

from llama_index.llms.base import (
    LLM,
    LLMMetadata,
    ChatDeltaResponse,
    ChatMessage,
    ChatResponse,
    ChatResponseType,
    CompletionDeltaResponse,
    CompletionResponse,
    CompletionResponseType,
    Message,
    StreamChatResponse,
    StreamCompletionResponse,
)
from llama_index.llms.generic_utils import (
    chat_response_to_completion_response,
    completion_response_to_chat_response,
    messages_to_prompt,
    prompt_to_messages,
)
from llama_index.llms.openai_utils import (
    completion_with_retry,
    is_chat_model,
    openai_modelname_to_contextsize,
    to_openai_message_dicts,
)


class OpenAI(LLM, BaseModel):
    model: str = Field("gpt-3.5-turbo")
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 10

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=openai_modelname_to_contextsize(self.model),
            num_output=self.max_tokens or -1,
        )

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        if self._is_chat_model:
            return self._chat(messages, **kwargs)
        else:
            # normalize input
            prompt = messages_to_prompt(messages)
            completion_response = self._complete(prompt, **kwargs)
            # normalize output
            return completion_response_to_chat_response(completion_response)

    def stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> StreamChatResponse:
        if self._is_chat_model:
            return self._stream_chat(messages, **kwargs)
        else:
            # normalize input
            prompt = messages_to_prompt(messages)
            completion_response = self._stream_complete(prompt, **kwargs)
            # normalize output
            return completion_response_to_chat_response(completion_response)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if self._is_chat_model:
            messages = prompt_to_messages(prompt)
            chat_response = self._chat(messages, **kwargs)
            return chat_response_to_completion_response(chat_response)
        else:
            return self._complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs: Any) -> StreamCompletionResponse:
        if self._is_chat_model:
            messages = prompt_to_messages(prompt)
            chat_response = self._stream_chat(messages, **kwargs)
            return chat_response_to_completion_response(chat_response)
        else:
            return self._stream_complete(prompt, **kwargs)

    @property
    def _is_chat_model(self) -> bool:
        return is_chat_model(self.model)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        model_kwargs = {
            **base_kwargs,
            **self.additional_kwargs,
        }
        return model_kwargs

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    def _chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponseType:
        if not self._is_chat_model:
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = completion_with_retry(
            is_chat_model=self._is_chat_model,
            max_retries=self.max_retries,
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        role = response["choices"][0]["message"]["role"]
        text = response["choices"][0]["message"]["content"]
        return ChatResponse(
            message=ChatMessage(
                role=role,
                content=text,
            ),
            raw=response,
        )

    def _stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseType:
        if not self._is_chat_model:
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        all_kwargs = self._get_all_kwargs(**kwargs)

        def gen() -> Generator[ChatDeltaResponse, None, None]:
            text = ""
            for response in completion_with_retry(
                is_chat_model=self._is_chat_model,
                max_retries=self.max_retries,
                messages=message_dicts,
                stream=True,
                **all_kwargs,
            ):
                role = response["choices"][0]["delta"].get("role", "assistant")
                delta = response["choices"][0]["delta"].get("content", "")
                text += delta
                yield ChatDeltaResponse(
                    message=ChatMessage(
                        role=role,
                        content=text,
                    ),
                    delta=delta,
                    raw=response,
                )

        return gen()

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponseType:
        if self._is_chat_model:
            raise ValueError("This model is a chat model.")

        all_kwargs = self._get_all_kwargs(**kwargs)
        if self.max_tokens is None:
            # NOTE: non-chat completion endpoint requires max_tokens to be set
            max_tokens = self._get_max_token_for_prompt(prompt)
            all_kwargs["max_tokens"] = max_tokens

        response = completion_with_retry(
            is_chat_model=self._is_chat_model,
            max_retries=self.max_retries,
            prompt=prompt,
            stream=False,
            **all_kwargs,
        )
        text = response["choices"][0]["text"]
        return CompletionResponse(
            text=text,
            raw=response,
        )

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseType:
        if self._is_chat_model:
            raise ValueError("This model is a chat model.")

        all_kwargs = self._get_all_kwargs(**kwargs)
        if self.max_tokens is None:
            # NOTE: non-chat completion endpoint requires max_tokens to be set
            max_tokens = self._get_max_token_for_prompt(prompt)
            all_kwargs["max_tokens"] = max_tokens

        def gen() -> Generator[CompletionDeltaResponse, None, None]:
            text = ""
            for response in completion_with_retry(
                is_chat_model=self._is_chat_model,
                max_retries=self.max_retries,
                prompt=prompt,
                stream=True,
                **all_kwargs,
            ):
                delta = response["choices"][0]["delta"].get("content", "")
                text += delta
                yield CompletionDeltaResponse(
                    delta=delta,
                    text=text,
                    raw=response,
                )

        return gen()

    def _get_max_token_for_prompt(self, prompt: str) -> int:
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Please install tiktoken to use the max_tokens=None feature."
            )
        context_window = self.metadata.context_window
        encoding = tiktoken.encoding_for_model(self.model)
        tokens = encoding.encode(prompt)
        max_token = context_window - len(tokens)
        if max_token <= 0:
            raise ValueError(
                f"The prompt is too long for the model. "
                f"Please use a prompt that is less than {context_window} tokens."
            )
        return max_token

    async def achat(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> ChatResponse:
        # TODO: implement async chat
        return self.chat(messages, **kwargs)

    async def astream_chat(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> StreamChatResponse:
        # TODO: implement async chat
        return self.stream_chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # TODO: implement async complete
        return self.complete(prompt, **kwargs)

    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> StreamCompletionResponse:
        # TODO: implement async complete
        return self.stream_complete(prompt, **kwargs)
