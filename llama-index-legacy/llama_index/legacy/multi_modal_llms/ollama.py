from typing import Any, Dict, Sequence, Tuple

from llama_index.legacy.bridge.pydantic import Field
from llama_index.legacy.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.legacy.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.legacy.multi_modal_llms.generic_utils import image_documents_to_base64
from llama_index.legacy.schema import ImageDocument


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
    model: str = Field(description="The MultiModal Ollama model to use.")
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
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model parameters for the Ollama API.",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Init params."""
        # make sure that ollama is installed
        try:
            import ollama  # noqa: F401
        except ImportError:
            raise ImportError(
                "Ollama is not installed. Please install it using `pip install ollama`."
            )
        super().__init__(**kwargs)

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
        import ollama

        ollama_messages = _messages_to_dicts(messages)
        response = ollama.chat(
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
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Stream chat."""
        import ollama

        ollama_messages = _messages_to_dicts(messages)
        response = ollama.chat(
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
        image_documents: Sequence[ImageDocument],
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Complete."""
        import ollama

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            images=image_documents_to_base64(image_documents),
            stream=False,
            options=self._model_kwargs,
        )
        return CompletionResponse(
            text=response["response"],
            raw=response,
            additional_kwargs=get_additional_kwargs(response, ("response",)),
        )

    def stream_complete(
        self,
        prompt: str,
        image_documents: Sequence[ImageDocument],
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """Stream complete."""
        import ollama

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            images=image_documents_to_base64(image_documents),
            stream=True,
            options=self._model_kwargs,
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
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError("Ollama does not support async completion.")

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError("Ollama does not support async chat.")

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError("Ollama does not support async streaming completion.")

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError("Ollama does not support async streaming chat.")
