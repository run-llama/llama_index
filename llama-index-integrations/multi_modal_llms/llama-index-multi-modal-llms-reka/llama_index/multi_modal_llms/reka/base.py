import os
from typing import Any, Dict, List, Optional, Sequence
from PIL import Image
import base64
import io

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
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.multi_modal_llms import MultiModalLLM, MultiModalLLMMetadata
from llama_index.core.schema import ImageDocument

try:
    from reka.client import Reka, AsyncReka
    from reka.core import ApiError
except ImportError:
    raise ValueError(
        "Reka is not installed. Please install it with `pip install reka-api`."
    )

DEFAULT_REKA_MODEL = "reka-flash"
DEFAULT_REKA_MAX_TOKENS = 512
DEFAULT_REKA_CONTEXT_WINDOW = 128000


def process_messages_for_reka(messages: Sequence[ChatMessage]) -> List[Dict[str, str]]:
    reka_messages = []
    system_message = None

    for message in messages:
        if message.role == MessageRole.SYSTEM:
            if system_message is None:
                system_message = message.content
            else:
                raise ValueError("Multiple system messages are not supported.")
        elif message.role == MessageRole.USER:
            content = message.content
            if system_message:
                content = f"{system_message}\n{content}"
                system_message = None
            reka_messages.append({"role": "user", "content": content})
        elif message.role == MessageRole.ASSISTANT:
            reka_messages.append({"role": "assistant", "content": message.content})
        else:
            raise ValueError(f"Unsupported message role: {message.role}")

    return reka_messages


class RekaMultiModalLLM(MultiModalLLM):
    """Reka Multi-Modal LLM integration for LlamaIndex."""

    model: str = Field(default=DEFAULT_REKA_MODEL, description="The Reka model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_REKA_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for Reka API calls.",
    )

    _client: Reka = PrivateAttr()
    _aclient: AsyncReka = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_REKA_MODEL,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_REKA_MAX_TOKENS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_key = api_key or os.getenv("REKA_API_KEY")
        if not api_key:
            raise ValueError(
                "Reka API key is required. Please provide it as an argument or set the REKA_API_KEY environment variable."
            )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
        )

        self._client = Reka(api_key=api_key)
        self._aclient = AsyncReka(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "Reka_MultiModal_LLM"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        return MultiModalLLMMetadata(
            context_window=DEFAULT_REKA_CONTEXT_WINDOW,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {**self._model_kwargs, **kwargs}

    def _process_images(
        self, image_documents: Sequence[ImageDocument]
    ) -> List[Dict[str, Any]]:
        image_contents = []
        for doc in image_documents:
            try:
                image_data = doc.resolve_image()
                if isinstance(image_data, str):
                    # It's a file path or URL
                    if image_data.startswith(("http://", "https://")):
                        image_contents.append(
                            {"type": "image_url", "image_url": image_data}
                        )
                    else:
                        # It's a local file path
                        with open(image_data, "rb") as image_file:
                            img = Image.open(image_file)
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            image_contents.append(
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{img_str}",
                                }
                            )
                elif isinstance(image_data, io.BytesIO):
                    # It's binary data
                    img = Image.open(image_data)
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    image_contents.append(
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_str}",
                        }
                    )
                else:
                    raise ValueError("Unsupported image data type")
            except Exception as e:
                raise ValueError(f"Failed to process image: {e!s}")

        return image_contents

    @llm_chat_callback()
    def chat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        if image_documents:
            image_contents = self._process_images(image_documents)
            reka_messages[-1]["content"] = [
                *image_contents,
                {"type": "text", "text": reka_messages[-1]["content"]},
            ]

        try:
            response = self._client.chat.create(messages=reka_messages, **all_kwargs)
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.responses[0].message.content,
                ),
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        content = []

        if image_documents:
            content.extend(self._process_images(image_documents))

        content.append({"type": "text", "text": prompt})

        try:
            response = self._client.chat.create(
                messages=[{"role": "user", "content": content}], **all_kwargs
            )
            return CompletionResponse(
                text=response.responses[0].message.content,
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_chat_callback()
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> ChatResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        if image_documents:
            image_contents = self._process_images(image_documents)
            reka_messages[-1]["content"] = [
                *image_contents,
                {"type": "text", "text": reka_messages[-1]["content"]},
            ]

        try:
            stream = self._client.chat.create_stream(
                messages=reka_messages, **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        def gen() -> ChatResponseGen:
            prev_content = ""
            for chunk in stream:
                content = chunk.responses[0].chunk.content
                content_delta = content[len(prev_content) :]
                prev_content = content
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=content,
                    ),
                    delta=content_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        content = []

        if image_documents:
            content.extend(self._process_images(image_documents))

        content.append({"type": "text", "text": prompt})

        try:
            stream = self._client.chat.create_stream(
                messages=[{"role": "user", "content": content}], **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        def gen() -> CompletionResponseGen:
            prev_text = ""
            for chunk in stream:
                text = chunk.responses[0].chunk.content
                text_delta = text[len(prev_text) :]
                prev_text = text
                yield CompletionResponse(
                    text=text,
                    delta=text_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        if image_documents:
            image_contents = self._process_images(image_documents)
            reka_messages[-1]["content"] = [
                *image_contents,
                {"type": "text", "text": reka_messages[-1]["content"]},
            ]

        try:
            response = await self._aclient.chat.create(
                messages=reka_messages, **all_kwargs
            )
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.responses[0].message.content,
                ),
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_completion_callback()
    async def acomplete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        content = []

        if image_documents:
            content.extend(self._process_images(image_documents))

        content.append({"type": "text", "text": prompt})

        try:
            response = await self._aclient.chat.create(
                messages=[{"role": "user", "content": content}], **all_kwargs
            )
            return CompletionResponse(
                text=response.responses[0].message.content,
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        if image_documents:
            image_contents = self._process_images(image_documents)
            reka_messages[-1]["content"] = [
                *image_contents,
                {"type": "text", "text": reka_messages[-1]["content"]},
            ]

        try:
            stream = self._aclient.chat.create_stream(
                messages=reka_messages, **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        async def gen() -> ChatResponseAsyncGen:
            prev_content = ""
            async for chunk in stream:
                content = chunk.responses[0].chunk.content
                content_delta = content[len(prev_content) :]
                prev_content = content
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=content,
                    ),
                    delta=content_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        content = []

        if image_documents:
            content.extend(self._process_images(image_documents))

        content.append({"type": "text", "text": prompt})

        try:
            stream = self._aclient.chat.create_stream(
                messages=[{"role": "user", "content": content}], **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        async def gen() -> CompletionResponseAsyncGen:
            prev_text = ""
            async for chunk in stream:
                text = chunk.responses[0].chunk.content
                text_delta = text[len(prev_text) :]
                prev_text = text
                yield CompletionResponse(
                    text=text,
                    delta=text_delta,
                    raw=chunk.__dict__,
                )

        return gen()
