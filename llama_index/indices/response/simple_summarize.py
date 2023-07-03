from typing import Any, Generator, Sequence, cast

from llama_index.indices.response.base_builder import BaseResponseBuilder
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.types import RESPONSE_TEXT_TYPE


class SimpleSummarize(BaseResponseBuilder):
    def __init__(
        self,
        service_context: ServiceContext,
        text_qa_template: QuestionAnswerPrompt,
        streaming: bool = False,
    ) -> None:
        super().__init__(service_context, streaming)
        self._text_qa_template = text_qa_template

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        truncated_chunks = self._service_context.prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=text_chunks,
        )
        node_text = "\n".join(truncated_chunks)

        response: RESPONSE_TEXT_TYPE
        if not self._streaming:
            response = await self._service_context.llm_predictor.apredict(
                text_qa_template,
                context_str=node_text,
            )
        else:
            response = self._service_context.llm_predictor.stream(
                text_qa_template,
                context_str=node_text,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        truncated_chunks = self._service_context.prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=text_chunks,
        )
        node_text = "\n".join(truncated_chunks)

        response: RESPONSE_TEXT_TYPE
        if not self._streaming:
            response = self._service_context.llm_predictor.predict(
                text_qa_template,
                context_str=node_text,
            )
        else:
            response = self._service_context.llm_predictor.stream(
                text_qa_template,
                context_str=node_text,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response
