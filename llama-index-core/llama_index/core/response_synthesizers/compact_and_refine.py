from typing import Any, List, Optional, Sequence

from llama_index.core.prompts.prompt_utils import get_biggest_prompt
from llama_index.core.response_synthesizers.refine import Refine
from llama_index.core.types import RESPONSE_TEXT_TYPE
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


class CompactAndRefine(Refine):
    """Refine responses across compact text chunks."""

    @dispatcher.span
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        compact_texts = self._make_compact_text_chunks(query_str, text_chunks)
        return await super().aget_response(
            query_str=query_str,
            text_chunks=compact_texts,
            prev_response=prev_response,
            **response_kwargs,
        )

    @dispatcher.span
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get compact response."""
        # use prompt helper to fix compact text_chunks under the prompt limitation
        # TODO: This is a temporary fix - reason it's temporary is that
        # the refine template does not account for size of previous answer.
        new_texts = self._make_compact_text_chunks(query_str, text_chunks)
        return super().get_response(
            query_str=query_str,
            text_chunks=new_texts,
            prev_response=prev_response,
            **response_kwargs,
        )

    def _make_compact_text_chunks(
        self, query_str: str, text_chunks: Sequence[str]
    ) -> List[str]:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        refine_template = self._refine_template.partial_format(query_str=query_str)

        max_prompt = get_biggest_prompt([text_qa_template, refine_template])
        return self._prompt_helper.repack(max_prompt, text_chunks)
