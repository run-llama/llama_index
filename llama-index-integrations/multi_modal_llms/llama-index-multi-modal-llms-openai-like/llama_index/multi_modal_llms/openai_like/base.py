from typing import Any, Optional, Sequence
from pathlib import Path

from llama_index.core.base.llms.generic_utils import (
    chat_response_to_completion_response,
    stream_chat_response_to_completion_response,
    astream_chat_response_to_completion_response,
)
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
    ImageBlock,
)
from llama_index.core.schema import ImageNode
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.llms.openai_like.base import OpenAILike


class OpenAILikeMultiModal(OpenAILike):
    """
    OpenAI-like Multi-Modal LLM.

    This class combines the multi-modal capabilities of OpenAIMultiModal with the
    flexibility of OpenAI-like, allowing you to use multi-modal features with
    third-party OpenAI-compatible APIs.

    Args:
        model (str):
            The model to use for the api.
        api_base (str):
            The base url to use for the api.
            Defaults to "https://api.openai.com/v1".
        is_chat_model (bool):
            Whether the model uses the chat or completion endpoint.
            Defaults to True for multi-modal models.
        is_function_calling_model (bool):
            Whether the model supports OpenAI function calling/tools over the API.
            Defaults to False.
        api_key (str):
            The api key to use for the api.
            Set this to some random string if your API does not require an api key.
        context_window (int):
            The context window to use for the api. Set this to your model's context window for the best experience.
            Defaults to 3900.
        max_tokens (int):
            The max number of tokens to generate.
            Defaults to None.
        temperature (float):
            The temperature to use for the api.
            Default is 0.1.
        additional_kwargs (dict):
            Specify additional parameters to the request body.
        max_retries (int):
            How many times to retry the API call if it fails.
            Defaults to 3.
        timeout (float):
            How long to wait, in seconds, for an API call before failing.
            Defaults to 60.0.
        reuse_client (bool):
            Reuse the OpenAI client between requests.
            Defaults to True.
        default_headers (dict):
            Override the default headers for API requests.
            Defaults to None.
        http_client (httpx.Client):
            Pass in your own httpx.Client instance.
            Defaults to None.
        async_http_client (httpx.AsyncClient):
            Pass in your own httpx.AsyncClient instance.
            Defaults to None.
        tokenizer (Union[Tokenizer, str, None]):
            An instance of a tokenizer object that has an encode method, or the name
            of a tokenizer model from Hugging Face. If left as None, then this
            disables inference of max_tokens.

    Examples:
        `pip install llama-index-llms-openai-like`

        ```python
        from llama_index.llms.openai_like import OpenAILikeMultiModal
        from llama_index.core.schema import ImageNode

        llm = OpenAILikeMultiModal(
            model="gpt-4-vision-preview",
            api_base="https://api.openai.com/v1",
            api_key="your-api-key",
            context_window=128000,
            is_chat_model=True,
            is_function_calling_model=False,
        )

        # Create image nodes
        image_nodes = [ImageNode(image_url="https://example.com/image.jpg")]

        # Complete with images
        response = llm.complete("Describe this image", image_documents=image_nodes)
        print(str(response))
        ```

    """

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=LLMMetadata.model_fields["context_window"].description,
    )
    is_chat_model: bool = Field(
        default=True,  # Default to True for multi-modal models
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=LLMMetadata.model_fields["is_function_calling_model"].description,
    )

    @classmethod
    def class_name(cls) -> str:
        return "openai_like_multi_modal_llm"

    def _get_multi_modal_chat_message(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[ImageNode],
        image_detail: Optional[str] = "low",
    ) -> ChatMessage:
        """Create a multi-modal chat message with text and images."""
        chat_msg = ChatMessage(role=role, content=prompt)
        if not image_documents:
            # if image_documents is empty, return text only chat message
            return chat_msg

        for image_document in image_documents:
            # Create the appropriate ContentBlock depending on the document content
            if image_document.image:
                chat_msg.blocks.append(
                    ImageBlock(
                        image=bytes(image_document.image, encoding="utf-8"),
                        detail=image_detail,
                    )
                )
            elif image_document.image_url:
                chat_msg.blocks.append(
                    ImageBlock(url=image_document.image_url, detail=image_detail)
                )
            elif image_document.image_path:
                chat_msg.blocks.append(
                    ImageBlock(
                        path=Path(image_document.image_path),
                        detail=image_detail,
                        image_mimetype=image_document.image_mimetype
                        or image_document.metadata.get("file_type"),
                    )
                )
            elif f_path := image_document.metadata.get("file_path"):
                chat_msg.blocks.append(
                    ImageBlock(
                        path=Path(f_path),
                        detail=image_detail,
                        image_mimetype=image_document.metadata.get("file_type"),
                    )
                )
        return chat_msg

    def complete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageNode]] = None,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Complete the prompt with optional image documents."""
        if image_documents:
            # Use multi-modal completion
            chat_message = self._get_multi_modal_chat_message(
                prompt=prompt,
                role=MessageRole.USER,
                image_documents=image_documents,
            )
            chat_response = self.chat([chat_message], **kwargs)
            return chat_response_to_completion_response(chat_response)
        else:
            # Use regular completion from parent class
            return super().complete(prompt, formatted=formatted, **kwargs)

    def stream_complete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageNode]] = None,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """Stream complete the prompt with optional image documents."""
        if image_documents:
            # Use multi-modal streaming completion
            chat_message = self._get_multi_modal_chat_message(
                prompt=prompt,
                role=MessageRole.USER,
                image_documents=image_documents,
            )
            chat_response = self.stream_chat([chat_message], **kwargs)
            return stream_chat_response_to_completion_response(chat_response)
        else:
            # Use regular streaming completion from parent class
            return super().stream_complete(prompt, formatted=formatted, **kwargs)

    # ===== Async Endpoints =====

    async def acomplete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageNode]] = None,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Async complete the prompt with optional image documents."""
        if image_documents:
            # Use multi-modal async completion
            chat_message = self._get_multi_modal_chat_message(
                prompt=prompt,
                role=MessageRole.USER,
                image_documents=image_documents,
            )
            chat_response = await self.achat([chat_message], **kwargs)
            return chat_response_to_completion_response(chat_response)
        else:
            # Use regular async completion from parent class
            return await super().acomplete(prompt, formatted=formatted, **kwargs)

    async def astream_complete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageNode]] = None,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        """Async stream complete the prompt with optional image documents."""
        if image_documents:
            # Use multi-modal async streaming completion
            chat_message = self._get_multi_modal_chat_message(
                prompt=prompt,
                role=MessageRole.USER,
                image_documents=image_documents,
            )
            chat_response = await self.astream_chat([chat_message], **kwargs)
            return astream_chat_response_to_completion_response(chat_response)
        else:
            # Use regular async streaming completion from parent class
            return await super().astream_complete(prompt, formatted=formatted, **kwargs)

    # ===== Multi-Modal Chat Methods =====

    def multi_modal_chat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Optional[Sequence[ImageNode]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat with multi-modal support."""
        if image_documents and messages:
            # Add images to the last user message
            last_message = messages[-1]
            if last_message.role == MessageRole.USER:
                enhanced_message = self._get_multi_modal_chat_message(
                    prompt=last_message.content or "",
                    role=last_message.role,
                    image_documents=image_documents,
                )
                # Replace the last message with the enhanced one
                return self.chat([*list(messages[:-1]), enhanced_message], **kwargs)

        # Fall back to regular chat
        return self.chat(messages, **kwargs)

    def multi_modal_stream_chat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Optional[Sequence[ImageNode]] = None,
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Stream chat with multi-modal support."""
        if image_documents and messages:
            # Add images to the last user message
            last_message = messages[-1]
            if last_message.role == MessageRole.USER:
                enhanced_message = self._get_multi_modal_chat_message(
                    prompt=last_message.content or "",
                    role=last_message.role,
                    image_documents=image_documents,
                )
                # Replace the last message with the enhanced one
                return self.stream_chat(
                    [*list(messages[:-1]), enhanced_message], **kwargs
                )

        # Fall back to regular stream chat
        return self.stream_chat(messages, **kwargs)

    async def amulti_modal_chat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Optional[Sequence[ImageNode]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat with multi-modal support."""
        if image_documents and messages:
            # Add images to the last user message
            last_message = messages[-1]
            if last_message.role == MessageRole.USER:
                enhanced_message = self._get_multi_modal_chat_message(
                    prompt=last_message.content or "",
                    role=last_message.role,
                    image_documents=image_documents,
                )
                # Replace the last message with the enhanced one
                return await self.achat(
                    [*list(messages[:-1]), enhanced_message], **kwargs
                )

        # Fall back to regular async chat
        return await self.achat(messages, **kwargs)

    async def amulti_modal_stream_chat(
        self,
        messages: Sequence[ChatMessage],
        image_documents: Optional[Sequence[ImageNode]] = None,
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async stream chat with multi-modal support."""
        if image_documents and messages:
            # Add images to the last user message
            last_message = messages[-1]
            if last_message.role == MessageRole.USER:
                enhanced_message = self._get_multi_modal_chat_message(
                    prompt=last_message.content or "",
                    role=last_message.role,
                    image_documents=image_documents,
                )
                # Replace the last message with the enhanced one
                return await self.astream_chat(
                    [*list(messages[:-1]), enhanced_message], **kwargs
                )

        # Fall back to regular async stream chat
        return await self.astream_chat(messages, **kwargs)
