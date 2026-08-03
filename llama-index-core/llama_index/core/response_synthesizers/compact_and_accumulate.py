from typing import Any, Sequence

from llama_index.core.base.llms.types import ChatMessage

from llama_index.core.response_synthesizers import Accumulate
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core.utils import temp_set_attrs


class CompactAndAccumulate(Accumulate):
    """Accumulate responses across compact text chunks."""

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        separator: str = "\n---------------------\n",
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get compact response."""
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)

        with temp_set_attrs(self._prompt_helper):
            new_texts = self._prompt_helper.repack(
                text_qa_template, text_chunks, llm=self._llm
            )

            return await super().aget_response(
                query_str=query_str,
                text_chunks=new_texts,
                separator=separator,
                **response_kwargs,
            )

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        separator: str = "\n---------------------\n",
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get compact response."""
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)

        with temp_set_attrs(self._prompt_helper):
            new_texts = self._prompt_helper.repack(
                text_qa_template, text_chunks, llm=self._llm
            )

            return super().get_response(
                query_str=query_str,
                text_chunks=new_texts,
                separator=separator,
                **response_kwargs,
            )

    async def aget_response_from_messages(
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        separator: str = "\n---------------------\n",
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        chat_content_qa_template = self._chat_content_qa_template.partial_format(
            query_str=query_str
        )

        with temp_set_attrs(self._chat_prompt_helper):
            new_chunks = self._chat_prompt_helper.repack(
                chat_content_qa_template, list(message_chunks), llm=self._llm
            )

            return await super().aget_response_from_messages(
                query_str=query_str,
                message_chunks=new_chunks,
                separator=separator,
                **response_kwargs,
            )

    def get_response_from_messages(
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        separator: str = "\n---------------------\n",
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        chat_content_qa_template = self._chat_content_qa_template.partial_format(
            query_str=query_str
        )

        with temp_set_attrs(self._chat_prompt_helper):
            new_chunks = self._chat_prompt_helper.repack(
                chat_content_qa_template, list(message_chunks), llm=self._llm
            )

            return super().get_response_from_messages(
                query_str=query_str,
                message_chunks=new_chunks,
                separator=separator,
                **response_kwargs,
            )
