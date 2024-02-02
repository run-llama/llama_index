from typing import Any, Sequence

from llama_index.response_synthesizers import Accumulate
from llama_index.types import RESPONSE_TEXT_TYPE
from llama_index.utils import temp_set_attrs


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
        # use prompt helper to fix compact text_chunks under the prompt limitation
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)

        with temp_set_attrs(self._service_context.prompt_helper):
            new_texts = self._service_context.prompt_helper.repack(
                text_qa_template, text_chunks
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
        # use prompt helper to fix compact text_chunks under the prompt limitation
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)

        with temp_set_attrs(self._service_context.prompt_helper):
            new_texts = self._service_context.prompt_helper.repack(
                text_qa_template, text_chunks
            )

            return super().get_response(
                query_str=query_str,
                text_chunks=new_texts,
                separator=separator,
                **response_kwargs,
            )
