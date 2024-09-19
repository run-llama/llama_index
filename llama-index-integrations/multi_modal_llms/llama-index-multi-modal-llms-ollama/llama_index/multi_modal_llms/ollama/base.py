from typing import Any, Dict, Optional, Sequence, Tuple

from ollama import Client, AsyncClient

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.core.multi_modal_llms.generic_utils import image_documents_to_base64
from llama_index.core.schema import ImageNode


def get_additional_kwargs(
    response: Dict[str, Any], exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    return {k: v for k, v in response.items() if k not in exclude}


def _messages_to_dicts(messages: Sequence[ChatMessage]) -> Sequence[Dict[str, Any]]:
    """Convert messages to dicts.

    For use in ollama API

    """
    results = []
    for message in messages:
        # TODO: just pass through the image arg for now.
        # TODO: have a consistent interface between multimodal models
        images = message.additional_kwargs.get("images")
        results.append(
            {
                "role": message.role.value,
                "content": message.content,
                "images": images,
            }
        )
    return results


class OllamaMultiModal(MultiModalLLM):
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted under.",
    )
    model: str = Field(description="The MultiModal Ollama model to use.")
    temperature: float = Field(
        default=0.75,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    request_timeout: Optional[float] = Field(
        description="The timeout for making http request to Ollama API server",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model parameters for the Ollama API.",
    )
    _client: Client = PrivateAttr()
    _aclient: AsyncClient = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        """Init params and ollama client."""
        super().__init__(**kwargs)
        self._client = Client(host=self.base_url, timeout=self.request_timeout)
        self._aclient = AsyncClient(host=self.base_url, timeout=self.request_timeout)

    @classmethod
    def class_name(cls) -> str:
        return "Ollama_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """LLM metadata."""
        return MultiModalLLMMetadata(
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

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat."""
        ollama_messages = _messages_to_dicts(messages)
        response = self._client.chat(
            model=self.model, messages=ollama_messages, stream=False, **kwargs
        )
        return ChatResponse(
            message=ChatMessage(
                content=response["message"]["content"],
                role=MessageRole(response["message"]["role"]),
                additional_kwargs=get_additional_kwargs(response, ("message",)),
            ),
            raw=response["message"],
            additional_kwargs=get_additional_kwargs(response, ("message",)),
        )

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Stream chat."""
        ollama_messages = _messages_to_dicts(messages)
        response = self._client.chat(
            model=self.model, messages=ollama_messages, stream=True, **kwargs
        )
        text = ""
        for chunk in response:
            if "done" in chunk and chunk["done"]:
                break
            message = chunk["message"]
            delta = message.get("content")
            text += delta
            yield ChatResponse(
                message=ChatMessage(
                    content=text,
                    role=MessageRole(message["role"]),
                    additional_kwargs=get_additional_kwargs(
                        message, ("content", "role")
                    ),
                ),
                delta=delta,
                raw=message,
                additional_kwargs=get_additional_kwargs(chunk, ("message",)),
            )

    def complete(
        self,
        prompt: str,
        image_documents: Sequence[ImageNode],
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Complete."""
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            images=image_documents_to_base64(image_documents),
            stream=False,
            options=self._model_kwargs,
            **kwargs,
        )
        return CompletionResponse(
            text=response["response"],
            raw=response,
            additional_kwargs=get_additional_kwargs(response, ("response",)),
        )

    def stream_complete(
        self,
        prompt: str,
        image_documents: Sequence[ImageNode],
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """Stream complete."""
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            images=image_documents_to_base64(image_documents),
            stream=True,
            options=self._model_kwargs,
            **kwargs,
        )
        text = ""
        for chunk in response:
            if "done" in chunk and chunk["done"]:
                break
            delta = chunk.get("response")
            text += delta
            yield CompletionResponse(
                text=str(chunk["response"]),
                delta=delta,
                raw=chunk,
                additional_kwargs=get_additional_kwargs(chunk, ("response",)),
            )

    async def acomplete(
        self,
        prompt: str,
        image_documents: Sequence[ImageNode],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Async complete."""
        response = await self._aclient.generate(
            model=self.model,
            prompt=prompt,
            images=image_documents_to_base64(image_documents),
            stream=False,
            options=self._model_kwargs,
            **kwargs,
        )
        return CompletionResponse(
            text=response["response"],
            raw=response,
            additional_kwargs=get_additional_kwargs(response, ("response",)),
        )

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat."""
        ollama_messages = _messages_to_dicts(messages)
        response = await self._aclient.chat(
            model=self.model, messages=ollama_messages, stream=False, **kwargs
        )
        return ChatResponse(
            message=ChatMessage(
                content=response["message"]["content"],
                role=MessageRole(response["message"]["role"]),
                additional_kwargs=get_additional_kwargs(response, ("message",)),
            ),
            raw=response["message"],
            additional_kwargs=get_additional_kwargs(response, ("message",)),
        )

    async def astream_complete(
        self,
        prompt: str,
        image_documents: Sequence[ImageNode],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        """Async stream complete."""
        async for chunk in self._aclient.generate(
            model=self.model,
            prompt=prompt,
            images=image_documents_to_base64(image_documents),
            stream=True,
            options=self._model_kwargs,
            **kwargs,
        ):
            if "done" in chunk and chunk["done"]:
                break
            delta = chunk.get("response")
            yield CompletionResponse(
                text=str(chunk["response"]),
                delta=delta,
                raw=chunk,
                additional_kwargs=get_additional_kwargs(chunk, ("response",)),
            )

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async stream chat."""
        ollama_messages = _messages_to_dicts(messages)
        async for chunk in self._aclient.chat(
            model=self.model, messages=ollama_messages, stream=True, **kwargs
        ):
            if "done" in chunk and chunk["done"]:
                break
            message = chunk["message"]
            delta = message.get("content")
            yield ChatResponse(
                message=ChatMessage(
                    content=message["content"],
                    role=MessageRole(message["role"]),
                    additional_kwargs=get_additional_kwargs(
                        message, ("content", "role")
                    ),
                ),
                delta=delta,
                raw=message,
                additional_kwargs=get_additional_kwargs(chunk, ("message",)),
            )
