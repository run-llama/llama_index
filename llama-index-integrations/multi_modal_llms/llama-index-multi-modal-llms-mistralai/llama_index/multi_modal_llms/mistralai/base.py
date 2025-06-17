from typing import Any, Dict, List, Sequence, Union

from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.base.llms.generic_utils import (
    chat_response_to_completion_response,
    stream_chat_response_to_completion_response,
    astream_chat_response_to_completion_response,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ImageBlock,
)
from llama_index.core.schema import ImageNode
from llama_index.llms.mistralai import MistralAI


class MistralAIMultiModal(MistralAI):
    def __init__(
        self,
        model: str = "pixtral-12b-2409",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "mistral_multi_modal_llm"

    def _get_credential_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            **kwargs,
        }

    def _get_multi_modal_chat_messages(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[Union[ImageBlock, ImageNode]],
    ) -> List[ChatMessage]:
        blocks = []
        if all(isinstance(doc, ImageNode) for doc in image_documents):
            for image_document in image_documents:
                blocks.append(
                    ImageBlock(
                        image=image_document.image,
                        path=image_document.image_path,
                        url=image_document.image_url,
                        image_mimetype=image_document.image_mimetype,
                    )
                )
        elif all(isinstance(doc, ImageBlock) for doc in image_documents):
            blocks.extend(image_documents)
        else:
            raise ValueError(
                "The input image_documents must be a list of either ImageBlock (preferred) or ImageNode objects"
            )
        blocks.append(TextBlock(text=prompt))
        return [ChatMessage(role=role, blocks=blocks)]

    def complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageBlock, ImageNode]],
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Produce a completion response from the LLM.

        Args:
            prompt (str): The prompt for the image description.
            image_documents (Sequence[Union[ImageNode, ImageBlock]]): The sequence of image documents (preferably as ImageBlock) to use as LLM input.
            **kwargs (Any): Keyword arguments.

        Returns:
            CompletionResponse: Completion response from the LLM.

        """
        messages = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER.value, image_documents=image_documents
        )
        chat_response = self.chat(messages=messages, **kwargs)
        return chat_response_to_completion_response(chat_response)

    def stream_complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """
        Stream a completion response from the LLM.

        Args:
            prompt (str): The prompt for the image description.
            image_documents (Sequence[Union[ImageNode, ImageBlock]]): The sequence of image documents (preferably as ImageBlock) to use as LLM input.
            **kwargs (Any): Keyword arguments.

        Returns:
            CompletionResponseGen: Generator for the completion response from the LLM.

        """
        messages = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER.value, image_documents=image_documents
        )
        chat_response = self.stream_chat(messages=messages, **kwargs)
        return stream_chat_response_to_completion_response(chat_response)

    async def acomplete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Asynchronously produce a completion response from the LLM.

        Args:
            prompt (str): The prompt for the image description.
            image_documents (Sequence[Union[ImageNode, ImageBlock]]): The sequence of image documents (preferably as ImageBlock) to use as LLM input.
            **kwargs (Any): Keyword arguments.

        Returns:
            CompletionResponse: Completion response from the LLM.

        """
        messages = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER.value, image_documents=image_documents
        )
        chat_response = await self.achat(messages=messages, **kwargs)
        return chat_response_to_completion_response(chat_response)

    async def astream_complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        """
        Asynchronously stream the completion response from the LLM.

        Args:
            prompt (str): The prompt for the image description.
            image_documents (Sequence[Union[ImageNode, ImageBlock]]): The sequence of image documents (preferably as ImageBlock) to use as LLM input.
            **kwargs (Any): Keyword arguments.

        Returns:
            CompletionResponseAsyncGen: An async generator for the completion response.

        """
        messages = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER.value, image_documents=image_documents
        )
        chat_response = await self.astream_chat(messages=messages, **kwargs)
        return astream_chat_response_to_completion_response(chat_response)
