import logging
from typing import Any, Generator, Optional, Sequence, cast, Type
from pydantic import BaseModel, Field
from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.indices.utils import truncate_text
from llama_index.prompts.default_prompt_selectors import (
    DEFAULT_TEXT_QA_PROMPT_SEL,
    DEFAULT_REFINE_PROMPT_SEL,
)
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt, Prompt
from llama_index.response.utils import get_response_text
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.types import RESPONSE_TEXT_TYPE
from llama_index.program.base_program import BasePydanticProgram
from llama_index.program.openai_program import OpenAIPydanticProgram
from llama_index.program.llm_program import LLMTextCompletionProgram
from llama_index.output_parsers.pydantic import PydanticOutputParser

logger = logging.getLogger(__name__)


class StructuredRefineResponse(BaseModel):
    """
    Used to answer a given query based on the provided context.

    Also indicates if the query was satisfied with the provided answer.
    """

    answer: str = Field(
        description="The answer for the given query, based on the context and not "
        "prior knowledge."
    )
    query_satisfied: bool = Field(
        description="True if there was enough context given to provide an answer "
        "that satisfies the query."
    )


class DefaultRefineProgram(BasePydanticProgram):
    """
    Runs the query on the LLM as normal and always returns the answer with
    query_satisfied=True. In effect, doesn't do any answer filtering.
    """

    def __init__(self, prompt: Prompt, llm_predictor: BaseLLMPredictor):
        self._prompt = prompt
        self._llm_predictor = llm_predictor

    @property
    def output_cls(self) -> Type[BaseModel]:
        return StructuredRefineResponse

    def __call__(self, *args: Any, **kwds: Any) -> StructuredRefineResponse:
        answer = self._llm_predictor.predict(
            self._prompt,
            **kwds,
        )
        return StructuredRefineResponse(answer=answer, query_satisfied=True)

    async def acall(self, *args: Any, **kwds: Any) -> StructuredRefineResponse:
        answer = await self._llm_predictor.apredict(
            self._prompt,
            **kwds,
        )
        return StructuredRefineResponse(answer=answer, query_satisfied=True)


class Refine(BaseSynthesizer):
    """Refine a response to a query across text chunks."""

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        streaming: bool = False,
        verbose: bool = False,
        structured_answer_filtering: bool = False,
    ) -> None:
        super().__init__(service_context=service_context, streaming=streaming)
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
        self._refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
        self._verbose = verbose
        self._structured_answer_filtering = structured_answer_filtering

        if self._streaming and self._structured_answer_filtering:
            raise ValueError(
                "Streaming not supported with structured answer filtering."
            )

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response over chunks."""
        prev_response_obj = cast(
            Optional[RESPONSE_TEXT_TYPE], response_kwargs.get("prev_response", None)
        )
        response: Optional[RESPONSE_TEXT_TYPE] = None
        for text_chunk in text_chunks:
            if prev_response_obj is None:
                # if this is the first chunk, and text chunk already
                # is an answer, then return it
                response = self._give_response_single(
                    query_str,
                    text_chunk,
                )
            else:
                response = self._refine_response_single(
                    prev_response_obj, query_str, text_chunk
                )
            prev_response_obj = response
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response

    def _get_program(self, prompt: Prompt) -> BasePydanticProgram:
        if self._structured_answer_filtering:
            try:
                return OpenAIPydanticProgram.from_defaults(
                    StructuredRefineResponse,
                    prompt=prompt,
                    llm=self._service_context.llm,
                    verbose=self._verbose,
                )
            except ValueError:
                output_parser = PydanticOutputParser(StructuredRefineResponse)
                return LLMTextCompletionProgram.from_defaults(
                    output_parser,
                    prompt=prompt,
                    llm=self._service_context.llm,
                    verbose=self._verbose,
                )
        else:
            return DefaultRefineProgram(
                prompt=prompt,
                llm_predictor=self._service_context.llm_predictor,
            )

    def _give_response_single(
        self,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response given a query and a corresponding text chunk."""
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        text_chunks = self._service_context.prompt_helper.repack(
            text_qa_template, [text_chunk]
        )

        response: Optional[RESPONSE_TEXT_TYPE] = None
        program = self._get_program(text_qa_template)
        # TODO: consolidate with loop in get_response_default
        for cur_text_chunk in text_chunks:
            query_satisfied = False
            if response is None and not self._streaming:
                structured_response = cast(
                    StructuredRefineResponse, program(context_str=cur_text_chunk)
                )
                query_satisfied = structured_response.query_satisfied
                if query_satisfied:
                    response = structured_response.answer
            elif response is None and self._streaming:
                response = self._service_context.llm_predictor.stream(
                    text_qa_template,
                    context_str=cur_text_chunk,
                )
                query_satisfied = True
            else:
                response = self._refine_response_single(
                    cast(RESPONSE_TEXT_TYPE, response),
                    query_str,
                    cur_text_chunk,
                )
        if response is None:
            response = "Empty Response"
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response

    def _refine_response_single(
        self,
        response: RESPONSE_TEXT_TYPE,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> Optional[RESPONSE_TEXT_TYPE]:
        """Refine response."""
        # TODO: consolidate with logic in response/schema.py
        if isinstance(response, Generator):
            response = get_response_text(response)

        fmt_text_chunk = truncate_text(text_chunk, 50)
        logger.debug(f"> Refine context: {fmt_text_chunk}")
        if self._verbose:
            print(f"> Refine context: {fmt_text_chunk}")
        # NOTE: partial format refine template with query_str and existing_answer here
        refine_template = self._refine_template.partial_format(
            query_str=query_str, existing_answer=response
        )
        text_chunks = self._service_context.prompt_helper.repack(
            refine_template, text_chunks=[text_chunk]
        )

        program = self._get_program(refine_template)
        for cur_text_chunk in text_chunks:
            query_satisfied = False
            if not self._streaming:
                structured_response = cast(
                    StructuredRefineResponse, program(context_msg=cur_text_chunk)
                )
                query_satisfied = structured_response.query_satisfied
                if query_satisfied:
                    response = structured_response.answer
            else:
                response = self._service_context.llm_predictor.stream(
                    refine_template,
                    context_msg=cur_text_chunk,
                )
                query_satisfied = True
            if query_satisfied:
                refine_template = self._refine_template.partial_format(
                    query_str=query_str, existing_answer=response
                )

        return response

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        prev_response_obj = cast(
            Optional[RESPONSE_TEXT_TYPE], response_kwargs.get("prev_response", None)
        )
        response: Optional[RESPONSE_TEXT_TYPE] = None
        for text_chunk in text_chunks:
            if prev_response_obj is None:
                # if this is the first chunk, and text chunk already
                # is an answer, then return it
                response = await self._agive_response_single(
                    query_str,
                    text_chunk,
                )
            else:
                response = await self._arefine_response_single(
                    prev_response_obj, query_str, text_chunk
                )
            prev_response_obj = response
        if response is None:
            response = "Empty Response"
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response

    async def _arefine_response_single(
        self,
        response: RESPONSE_TEXT_TYPE,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> Optional[RESPONSE_TEXT_TYPE]:
        """Refine response."""
        # TODO: consolidate with logic in response/schema.py
        if isinstance(response, Generator):
            response = get_response_text(response)

        fmt_text_chunk = truncate_text(text_chunk, 50)
        logger.debug(f"> Refine context: {fmt_text_chunk}")
        # NOTE: partial format refine template with query_str and existing_answer here
        refine_template = self._refine_template.partial_format(
            query_str=query_str, existing_answer=response
        )
        text_chunks = self._service_context.prompt_helper.repack(
            refine_template, text_chunks=[text_chunk]
        )

        program = self._get_program(refine_template)
        for cur_text_chunk in text_chunks:
            query_satisfied = False
            if not self._streaming:
                structured_response = await program.acall(context_msg=cur_text_chunk)
                structured_response = cast(
                    StructuredRefineResponse, structured_response
                )
                query_satisfied = structured_response.query_satisfied
                if query_satisfied:
                    response = structured_response.answer
            else:
                raise ValueError("Streaming not supported for async")

            if query_satisfied:
                refine_template = self._refine_template.partial_format(
                    query_str=query_str, existing_answer=response
                )

        return response

    async def _agive_response_single(
        self,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response given a query and a corresponding text chunk."""
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        text_chunks = self._service_context.prompt_helper.repack(
            text_qa_template, [text_chunk]
        )

        response: Optional[RESPONSE_TEXT_TYPE] = None
        program = self._get_program(text_qa_template)
        # TODO: consolidate with loop in get_response_default
        for cur_text_chunk in text_chunks:
            if response is None and not self._streaming:
                structured_response = await program.acall(context_str=cur_text_chunk)
                structured_response = cast(
                    StructuredRefineResponse, structured_response
                )
                query_satisfied = structured_response.query_satisfied
                if query_satisfied:
                    response = structured_response.answer
            elif response is None and self._streaming:
                raise ValueError("Streaming not supported for async")
            else:
                response = await self._arefine_response_single(
                    cast(RESPONSE_TEXT_TYPE, response),
                    query_str,
                    cur_text_chunk,
                )
        if response is None:
            response = "Empty Response"
        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)
        return response
