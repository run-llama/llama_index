from typing import Any, List, Sequence, Union
from deprecated import deprecated

from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    ImageBlock,
    MessageRole,
    ChatMessage,
)
from llama_index.core.base.llms.generic_utils import (
    chat_response_to_completion_response,
    stream_chat_response_to_completion_response,
    astream_chat_response_to_completion_response,
)
from llama_index.core.constants import (
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.multi_modal_llms import (
    MultiModalLLMMetadata,
)
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.anthropic.utils import (
    generate_anthropic_multi_modal_chat_message,
)

from llama_index.llms.anthropic import Anthropic


@deprecated(
    reason="This class is deprecated and will be no longer maintained, use Anthropic from llama-index-llms-anthropic instead. See Multi Modal LLMs documentation for a complete guide on migration: https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/#multi-modal-llms",
    version="0.3.2",
)
class AnthropicMultiModal(Anthropic):
    """Anthropic Multi Modal LLM."""

    @classmethod
    def class_name(cls) -> str:
        return "anthropic_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            num_output=self.max_tokens or DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    def _get_multi_modal_chat_messages(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> List[ChatMessage]:
        return generate_anthropic_multi_modal_chat_message(
            prompt=prompt,
            role=role,
            image_documents=image_documents,
        )

    def _complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Complete the prompt with image support and optional tool calls."""
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )

        response = super().chat(
            messages=message_dict,
            **kwargs,
        )

        return chat_response_to_completion_response(chat_response=response)

    def _stream_complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseGen:
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )
        message_dict.insert(
            0,
            self._get_multi_modal_chat_messages(
                prompt=self.system_prompt,
                role=MessageRole.SYSTEM,
            )[0],
        )

        gen = super().stream_chat(messages=message_dict)

        return stream_chat_response_to_completion_response(chat_response_gen=gen)

    def complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        return self._complete(prompt, image_documents, **kwargs)

    def stream_complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseGen:
        return self._stream_complete(prompt, image_documents, **kwargs)

    # ===== Async Endpoints =====

    async def _acomplete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )
        message_dict.insert(
            0,
            self._get_multi_modal_chat_messages(
                prompt=self.system_prompt,
                role=MessageRole.SYSTEM,
            )[0],
        )
        response = await super().achat(messages=message_dict, **kwargs)

        return chat_response_to_completion_response(response)

    async def acomplete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        return await self._acomplete(prompt, image_documents, **kwargs)

    async def _astream_complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )
        message_dict.insert(
            0,
            self._get_multi_modal_chat_messages(
                prompt=self.system_prompt,
                role=MessageRole.SYSTEM,
            )[0],
        )

        gen = await super().astream_chat(
            messages=message_dict,
            **kwargs,
        )

        return astream_chat_response_to_completion_response(gen)

    async def astream_complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        return await self._astream_complete(prompt, image_documents, **kwargs)
