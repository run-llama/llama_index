import json
from typing import Any, Dict, Sequence, Tuple

import httpx
from httpx import Timeout

from llama_index.bridge.pydantic import Field
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.llms.custom import CustomLLM

DEFAULT_REQUEST_TIMEOUT = 30.0


def get_addtional_kwargs(
    response: Dict[str, Any], exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    return {k: v for k, v in response.items() if k not in exclude}


class Ollama(CustomLLM):
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted under.",
    )
    model: str = Field(description="The Ollama model to use.")
    temperature: float = Field(
        default=0.75,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to Ollama API server",
    )
    prompt_key: str = Field(
        default="prompt", description="The key to use for the prompt in API calls."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model parameters for the Ollama API.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "Ollama_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=True,  # Ollama supports chat API for all models
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "num_ctx": self.context_window,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    **message.additional_kwargs,
                }
                for message in messages
            ],
            "options": self._model_kwargs,
            "stream": False,
            **kwargs,
        }

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
            message = raw["message"]
            return ChatResponse(
                message=ChatMessage(
                    content=message.get("content"),
                    role=MessageRole(message.get("role")),
                    additional_kwargs=get_addtional_kwargs(
                        message, ("content", "role")
                    ),
                ),
                raw=raw,
                additional_kwargs=get_addtional_kwargs(raw, ("message",)),
            )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    **message.additional_kwargs,
                }
                for message in messages
            ],
            "options": self._model_kwargs,
            "stream": True,
            **kwargs,
        }

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            with client.stream(
                method="POST",
                url=f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                text = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        message = chunk["message"]
                        delta = message.get("content")
                        text += delta
                        yield ChatResponse(
                            message=ChatMessage(
                                content=text,
                                role=MessageRole(message.get("role")),
                                additional_kwargs=get_addtional_kwargs(
                                    message, ("content", "role")
                                ),
                            ),
                            delta=delta,
                            raw=chunk,
                            additional_kwargs=get_addtional_kwargs(chunk, ("message",)),
                        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        payload = {
            self.prompt_key: prompt,
            "model": self.model,
            "options": self._model_kwargs,
            "stream": False,
            **kwargs,
        }

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
            text = raw.get("response")
            return CompletionResponse(
                text=text,
                raw=raw,
                additional_kwargs=get_addtional_kwargs(raw, ("response",)),
            )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        payload = {
            self.prompt_key: prompt,
            "model": self.model,
            "options": self._model_kwargs,
            "stream": True,
            **kwargs,
        }

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            with client.stream(
                method="POST",
                url=f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                text = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        delta = chunk.get("response")
                        text += delta
                        yield CompletionResponse(
                            delta=delta,
                            text=text,
                            raw=chunk,
                            additional_kwargs=get_addtional_kwargs(
                                chunk, ("response",)
                            ),
                        )
