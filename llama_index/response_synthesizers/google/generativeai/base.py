import logging
from typing import Any, List, Sequence, cast

from llama_index.bridge.pydantic import BaseModel  # type: ignore
from llama_index.prompts.mixin import PromptDictType
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.types import RESPONSE_TEXT_TYPE
from llama_index.vector_stores.google.generativeai import google_service_context

_logger = logging.getLogger(__name__)
_import_err_msg = "`google.generativeai` package not found, please run `pip install google-generativeai`"
_separator = "\n\n"


class SynthesizedResponse(BaseModel):
    answer: str
    attributed_passages: List[str]
    answerable_probability: float


class GoogleTextSynthesizer(BaseSynthesizer):
    _client: Any
    _answer_style: int

    def __init__(self, answer_style: int = 1, **kwargs: Any):
        try:
            import google.ai.generativelanguage as genai

            import llama_index.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        super().__init__(
            service_context=google_service_context,
            output_cls=SynthesizedResponse,
            **kwargs,
        )
        self._client = genaix.build_text_service()
        self._answer_style = genai.AnswerStyle(answer_style)

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> SynthesizedResponse:
        try:
            import google.ai.generativelanguage as genai

            import llama_index.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        _logger.debug(
            f"""\n
GoogleTextSynthesizer.get_response(
    query_str="{query_str}",
    text_chunks=[
{_separator.join(text_chunks)}
    ],
    {response_kwargs})"""
        )

        client = cast(genai.TextServiceClient, self._client)
        response = genaix.generate_text_answer(
            prompt=query_str,
            passages=list(text_chunks),
            answer_style=genai.AnswerStyle(self._answer_style),
            client=client,
        )

        return SynthesizedResponse(
            answer=response.answer,
            attributed_passages=[
                passage.text for passage in response.attributed_passages
            ],
            answerable_probability=response.answerable_probability,
        )

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return self.get_response(query_str, text_chunks, **response_kwargs)

    def _get_prompts(self) -> PromptDictType:
        # Not used.
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        # Not used.
        ...
