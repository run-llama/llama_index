from typing import Any, Optional, Sequence
from llama_index.core.base.llms.types import (
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
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
import requests
import os


class Maritalk(LLM):
    api_key: Optional[str] = Field(default=None, description="Your MariTalk API key.")
    temperature: float = Field(
        default=0.7,
        gt=0.0,
        lt=1.0,
        description="Run inference with this temperature. Must be in the"
        "closed interval [0.0, 1.0].",
    )
    max_tokens: int = Field(
        default=512,
        gt=0,
        description="The maximum number of tokens to" "generate in the reply.",
    )
    do_sample: bool = Field(
        default=True,
        description="Whether or not to use sampling; use `True` to enable.",
    )
    top_p: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Nucleus sampling parameter controlling the size of"
        " the probability mass considered for sampling.",
    )

    _endpoint: str = PrivateAttr("https://chat.maritaca.ai/api/chat/inference")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # If an API key is not provided during instantiation,
        # fall back to the MARITALK_API_KEY environment variable
        self.api_key = self.api_key or os.getenv("MARITALK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "An API key must be provided or set in the "
                "'MARITALK_API_KEY' environment variable."
            )

    @classmethod
    def class_name(cls) -> str:
        return "Maritalk"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name="maritalk",
            context_window=self.max_tokens,
            is_chat_model=True,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Prepare the data payload for the Maritalk API
        formatted_messages = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Add system message as a user message
                formatted_messages.append({"role": "user", "content": msg.content})
                # Follow it by an assistant message acknowledging it, to maintain conversation flow
                formatted_messages.append({"role": "assistant", "content": "ok"})
            else:
                # Format user and assistant messages as before
                formatted_messages.append(
                    {
                        "role": "user" if msg.role == MessageRole.USER else "assistant",
                        "content": msg.content,
                    }
                )

        data = {
            "messages": formatted_messages,
            "do_sample": self.do_sample,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        # Update data payload with additional kwargs if any
        data.update(kwargs)

        headers = {"authorization": f"Key {self.api_key}"}

        response = requests.post(self._endpoint, json=data, headers=headers)
        if response.status_code == 429:
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="Rate limited, please try again soon",
                ),
                raw=response.text,
            )
        elif response.ok:
            answer = response.json()["answer"]
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=answer),
                raw=response.json(),
            )
        else:
            response.raise_for_status()  # noqa: RET503

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        # Prepare the data payload for the Maritalk API
        data = {
            "messages": prompt,
            "do_sample": self.do_sample,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "chat_mode": False,
        }

        # Update data payload with additional kwargs if any
        data.update(kwargs)

        headers = {"authorization": f"Key {self.api_key}"}

        response = requests.post(self._endpoint, json=data, headers=headers)
        if response.status_code == 429:
            return CompletionResponse(
                text="Rate limited, please try again soon",
                raw=response.text,
            )
        elif response.ok:
            answer = response.json()["answer"]
            return CompletionResponse(
                text=answer,
                raw=response.json(),
            )
        else:
            response.raise_for_status()  # noqa: RET503

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError(
            "Maritalk does not currently support streaming completion."
        )

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError(
            "Maritalk does not currently support streaming completion."
        )

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, formatted, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError(
            "Maritalk does not currently support streaming completion."
        )

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError(
            "Maritalk does not currently support streaming completion."
        )
