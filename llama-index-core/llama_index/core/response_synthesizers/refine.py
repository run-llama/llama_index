import logging
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Optional,
    Sequence,
    Type,
    cast,
)

from llama_index.core.base.llms.types import ChatMessage

from llama_index.core.bridge.pydantic import BaseModel, Field, ValidationError
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.prompts.chat_prompts import (
    CHAT_CONTENT_QA_PROMPT,
    CHAT_CONTENT_REFINE_PROMPT,
)
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_REFINE_PROMPT_SEL,
    DEFAULT_TEXT_QA_PROMPT_SEL,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.prompts.prompt_utils import get_biggest_prompt
from llama_index.core.response.utils import get_response_text, aget_response_text
from llama_index.core.response_synthesizers.base import (
    BaseSynthesizer,
)
from llama_index.core.types import RESPONSE_TEXT_TYPE, BasePydanticProgram
from llama_index.core.instrumentation.events.synthesis import (
    GetResponseEndEvent,
    GetResponseStartEvent,
    GetMessageResponseEndEvent,
    GetMessageResponseStartEvent,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)

logger = logging.getLogger(__name__)

DEFAULT_RESPONSE_PADDING_SIZE = 500


class StructuredRefineResponse(BaseModel):
    """
    Used to answer a given query based on the provided context.

    Also indicates if the query was satisfied with the provided answer.
    """

    # It is important that this variable comes first, because the LLM *almost always* streams jsons in the order the
    # variables are listed here (and consequently serialized to json). What this allows is a very early abandonment
    # of irrelevant sources during streaming. Since this variable is streamed with in the first 10 or so tokens,
    # if it is False, we do not need to wait for the rest of the response, we can simply move on to the next source.
    query_satisfied: bool = Field(
        description="True if there was enough context given to provide an answer "
        "that satisfies the query."
    )
    answer: str = Field(
        description="The answer for the given query, based on the context and not "
        "prior knowledge."
    )


if TYPE_CHECKING:
    from llama_index.core.program.utils import create_flexible_model

    FlexibleStructuredRefineResponse = create_flexible_model(StructuredRefineResponse)


class DefaultRefineProgram(BasePydanticProgram):
    """
    Runs the query on the LLM as normal and always returns the answer with
    query_satisfied=True. In effect, doesn't do any answer filtering.
    """

    def __init__(
        self,
        prompt: BasePromptTemplate,
        llm: LLM,
        output_cls: Optional[Type[BaseModel]] = None,
    ):
        self._prompt = prompt
        self._llm = llm
        self._output_cls = output_cls

    @property
    def output_cls(self) -> Type[BaseModel]:
        return StructuredRefineResponse

    def __call__(self, *args: Any, **kwds: Any) -> StructuredRefineResponse:
        if self._output_cls is not None:
            answer = self._llm.structured_predict(
                self._output_cls,
                self._prompt,
                **kwds,
            )
            if isinstance(answer, BaseModel):
                answer = answer.model_dump_json()
        else:
            answer = self._llm.predict(
                self._prompt,
                **kwds,
            )
        return StructuredRefineResponse(answer=answer, query_satisfied=True)

    async def acall(self, *args: Any, **kwds: Any) -> StructuredRefineResponse:
        if self._output_cls is not None:
            answer = await self._llm.astructured_predict(  # type: ignore
                self._output_cls,
                self._prompt,
                **kwds,
            )
            if isinstance(answer, BaseModel):
                answer = answer.model_dump_json()
        else:
            answer = await self._llm.apredict(
                self._prompt,
                **kwds,
            )
        return StructuredRefineResponse(answer=answer, query_satisfied=True)

    def stream_call(
        self, *args: Any, **kwds: Any
    ) -> Generator[StructuredRefineResponse, None, None]:
        """
        Stream object.

        Returns a generator returning partials of the same object
        or a list of objects until it returns.
        """
        if self._output_cls is not None:
            for structured_answer in self._llm.stream_structured_predict(
                self._output_cls,
                self._prompt,
                **kwds,
            ):
                if answer := structured_answer.answer:  # type: ignore[union-attr]
                    yield StructuredRefineResponse(answer=answer, query_satisfied=True)
        else:
            answer = ""
            # Because structured stream_structured_predict does not yield partial json fields, answer is only available
            # once the field is complete. We want to mimic that behavior here so it behaves similarly across the two
            # cases
            for token in self._llm.stream(
                self._prompt,
                **kwds,
            ):
                answer += token
            yield StructuredRefineResponse(answer=answer.strip(), query_satisfied=True)

    async def astream_call(
        self, *args: Any, **kwds: Any
    ) -> AsyncGenerator[StructuredRefineResponse, None]:
        """
        Stream objects.

        Returns a generator returning partials of the same object
        or a list of objects until it returns.
        """

        async def gen() -> AsyncGenerator[StructuredRefineResponse, None]:
            if self._output_cls is not None:
                async for structured_answer in self._llm.astream_structured_predict(  # type: ignore
                    self._output_cls,
                    self._prompt,
                    **kwds,
                ):
                    if structured_answer.answer:
                        yield StructuredRefineResponse(
                            answer=structured_answer.answer, query_satisfied=True
                        )
            else:
                answer = ""
                # Because structured stream_structured_predict does not yield partial json fields, answer is only available
                # once the field is complete. We want to mimic that behavior here so it behaves similarly across the two
                # cases
                async for token in await self._llm.astream(
                    self._prompt,
                    **kwds,
                ):
                    answer += token
                if answer:
                    yield StructuredRefineResponse(
                        answer=answer.strip(), query_satisfied=True
                    )

        return gen()


class Refine(BaseSynthesizer):
    """Refine a response to a query across text chunks."""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        chat_content_qa_template: Optional[BasePromptTemplate] = None,
        chat_content_refine_template: Optional[BasePromptTemplate] = None,
        output_cls: Optional[Type[BaseModel]] = None,
        response_padding_size: int = DEFAULT_RESPONSE_PADDING_SIZE,
        streaming: bool = False,
        verbose: bool = False,
        structured_answer_filtering: bool = False,
        program_factory: Optional[
            Callable[[BasePromptTemplate], BasePydanticProgram]
        ] = None,
        multimodal: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            streaming=streaming,
            multimodal=multimodal,
        )
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
        self._refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
        self._chat_content_qa_template = (
            chat_content_qa_template or CHAT_CONTENT_QA_PROMPT
        )
        self._chat_content_refine_template = (
            chat_content_refine_template or CHAT_CONTENT_REFINE_PROMPT
        )
        self._verbose = verbose
        self._structured_answer_filtering = structured_answer_filtering
        self._output_cls = output_cls
        self._response_padding_size = response_padding_size

        if not self._structured_answer_filtering and program_factory is not None:
            raise ValueError(
                "Program factory not supported without structured answer filtering."
            )
        self._program_factory = program_factory or self._default_program_factory

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "text_qa_template": self._text_qa_template,
            "refine_template": self._refine_template,
            "chat_content_qa_template": self._chat_content_qa_template,
            "chat_content_refine_template": self._chat_content_refine_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_qa_template" in prompts:
            self._text_qa_template = prompts["text_qa_template"]
        if "refine_template" in prompts:
            self._refine_template = prompts["refine_template"]
        if "chat_content_qa_template" in prompts:
            self._chat_content_qa_template = prompts["chat_content_qa_template"]
        if "chat_content_refine_template" in prompts:
            self._chat_content_refine_template = prompts["chat_content_refine_template"]

    @staticmethod
    def _get_attribute_from_object_generator(
        generator: Generator, structured_response: BaseModel | None, attribute: str
    ) -> Generator:
        """
        Object generators like those returned by the DefaultRefineProgram or FunctionCallingProgram
        stream_call may yield multiple objects, but because we cannot guarantee the order of object attribute generation
        we need to wait until it's fully generated to be sure that the attribute is both present and complete
        """
        for obj in generator:
            structured_response = obj
        yield getattr(structured_response, attribute)

    @staticmethod
    async def _get_attribute_from_object_async_generator(
        generator: AsyncGenerator, structured_response: BaseModel | None, attribute: str
    ) -> AsyncGenerator:
        """
        Object generators like those returned by the DefaultRefineProgram or FunctionCallingProgram
        stream_call may yield multiple objects, but because we cannot guarantee the order of object attribute generation
        we need to wait until it's fully generated to be sure that the attribute is both present and complete
        """
        async for obj in generator:
            structured_response = obj
        yield getattr(structured_response, attribute)
        return

    def _default_program_factory(
        self, prompt: BasePromptTemplate
    ) -> BasePydanticProgram:
        if self._structured_answer_filtering:
            from llama_index.core.program.utils import get_program_for_llm

            return get_program_for_llm(
                StructuredRefineResponse,
                prompt,
                self._llm,
                verbose=self._verbose,
            )
        else:
            return DefaultRefineProgram(
                prompt=prompt,
                llm=self._llm,
                output_cls=self._output_cls,
            )

    def _update_response(
        self, program: BasePydanticProgram, program_kwargs: dict, response_kwargs: dict
    ) -> Optional[RESPONSE_TEXT_TYPE]:
        """Update response."""
        query_satisfied = False
        if not self._streaming:
            try:
                structured_response = cast(
                    StructuredRefineResponse,
                    program(
                        **program_kwargs,
                        **response_kwargs,
                    ),
                )
                query_satisfied = structured_response.query_satisfied
                if query_satisfied:
                    return structured_response.answer
            except (ValidationError, ValueError, TypeError) as e:
                logger.warning(f"Structured response error: {e}", exc_info=True)
        elif self._streaming:
            try:
                structured_response_gen = program.stream_call(
                    **program_kwargs,
                    **response_kwargs,
                )
                structured_response = None
                for sr in structured_response_gen:
                    structured_response = sr  # type: ignore[assignment]
                    if sr is not None:
                        query_satisfied = sr.query_satisfied  # type: ignore[union-attr]
                        if query_satisfied is not None:
                            break
                if query_satisfied:
                    return self._get_attribute_from_object_generator(
                        structured_response_gen,
                        structured_response,
                        "answer",  # type: ignore[arg-type]
                    )
            except (ValidationError, ValueError, TypeError) as e:
                logger.warning(f"Structured response error: {e}", exc_info=True)
        return None

    async def _aupdate_response(
        self, program: BasePydanticProgram, program_kwargs: dict, response_kwargs: dict
    ) -> Optional[RESPONSE_TEXT_TYPE]:
        """Update response."""
        query_satisfied = False
        if not self._streaming:
            try:
                structured_response = cast(
                    StructuredRefineResponse,
                    await program.acall(
                        **program_kwargs,
                        **response_kwargs,
                    ),
                )
                query_satisfied = structured_response.query_satisfied
                if query_satisfied:
                    return structured_response.answer
            except (ValidationError, ValueError, TypeError) as e:
                logger.warning(f"Structured response error: {e}", exc_info=True)
        elif self._streaming:
            try:
                structured_response_gen = await program.astream_call(
                    **program_kwargs,
                    **response_kwargs,
                )
                structured_response = None
                async for sr in structured_response_gen:
                    structured_response = sr  # type: ignore[assignment]
                    if sr is not None:
                        query_satisfied = sr.query_satisfied  # type: ignore[union-attr]
                        if query_satisfied is not None:
                            break
                if query_satisfied:
                    return self._get_attribute_from_object_async_generator(
                        structured_response_gen,
                        structured_response,
                        "answer",  # type: ignore[arg-type]
                    )
            except (ValidationError, ValueError, TypeError) as e:
                logger.warning(f"Structured response error: {e}", exc_info=True)
        return None

    def _run_refine_loop(
        self,
        query_str: str,
        chunks: Sequence[Any],
        qa_template: BasePromptTemplate,
        refine_template: BasePromptTemplate,
        make_qa_prompt_kwargs: Callable[[Any], dict],
        make_refine_prompt_kwargs: Callable[[Any], dict],
        start_event: Any,
        end_event: Any,
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        dispatcher.event(start_event)
        max_prompt = get_biggest_prompt([qa_template, refine_template])
        # Make first best guess at how many chunks we can fit in the prompt at once. Increase padding
        # to give more room for the response
        chunks_deque: deque = deque(
            [
                tc
                for chunk in chunks
                for tc in self._prompt_helper.repack(
                    max_prompt,
                    [chunk],  # type: ignore[list-item]
                    llm=self._llm,
                    padding=self._response_padding_size,
                )
            ]
        )
        response = prev_response
        while chunks_deque:
            if isinstance(response, Generator):
                response = get_response_text(response)

            chunk = chunks_deque.popleft()

            if response is None:
                prompt_template = qa_template.partial_format(query_str=query_str)
                prompt_kwargs = make_qa_prompt_kwargs(chunk)
            else:
                prompt_template = refine_template.partial_format(
                    query_str=query_str, existing_answer=response
                )
                # Because the existing answer portion is constantly being updated, it may be necessary to
                # repack the chunk with the new prompt template to ensure it fits. Since the template has been
                # partially formatted with the actual response, there's no need for the extra padding.
                repacked = self._prompt_helper.repack(
                    prompt_template,
                    [chunk],
                    llm=self._llm,  # type: ignore[list-item]
                )
                # If chunk is too big to be packed into a single chunk, push new chunks into the front of the deque
                if len(repacked) > 1:
                    chunks_deque.extendleft(repacked)
                    continue
                chunk = repacked[0]
                prompt_kwargs = make_refine_prompt_kwargs(chunk)

            program = self._program_factory(prompt_template)
            if resp := self._update_response(program, prompt_kwargs, response_kwargs):
                response = resp

        if isinstance(response, str):
            if self._output_cls is not None:
                try:
                    response = self._output_cls.model_validate_json(response)
                except (ValidationError, ValueError, TypeError):
                    pass
            else:
                response = response or "Empty Response"
        elif response is None:
            response = "Empty Response"
        else:
            response = cast(Generator, response)
        dispatcher.event(end_event)
        return response

    async def _arun_refine_loop(
        self,
        query_str: str,
        chunks: Sequence[Any],
        qa_template: BasePromptTemplate,
        refine_template: BasePromptTemplate,
        make_qa_prompt_kwargs: Callable[[Any], dict],
        make_refine_prompt_kwargs: Callable[[Any], dict],
        start_event: Any,
        end_event: Any,
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        dispatcher.event(start_event)
        max_prompt = get_biggest_prompt([qa_template, refine_template])
        # Make first best guess at how many chunks we can fit in the prompt at once. Increase padding
        # to give more room for the response
        chunks_deque: deque = deque(
            [
                tc
                for chunk in chunks
                for tc in self._prompt_helper.repack(
                    max_prompt,
                    [chunk],  # type: ignore[list-item]
                    llm=self._llm,
                    padding=self._response_padding_size,
                )
            ]
        )
        response = prev_response
        while chunks_deque:
            if isinstance(response, AsyncGenerator):
                response = await aget_response_text(response)

            chunk = chunks_deque.popleft()

            if response is None:
                prompt_template = qa_template.partial_format(query_str=query_str)
                prompt_kwargs = make_qa_prompt_kwargs(chunk)
            else:
                prompt_template = refine_template.partial_format(
                    query_str=query_str, existing_answer=response
                )
                # Because the existing answer portion is constantly being updated, it may be necessary to
                # repack the chunk with the new prompt template to ensure it fits. Since the template has been
                # partially formatted with the actual response, there's no need for the extra padding.
                repacked = self._prompt_helper.repack(
                    prompt_template,
                    [chunk],
                    llm=self._llm,  # type: ignore[list-item]
                )
                # If chunk is too big to be packed into a single chunk, push new chunks into the front of the deque
                if len(repacked) > 1:
                    chunks_deque.extendleft(repacked)
                    continue
                chunk = repacked[0]
                prompt_kwargs = make_refine_prompt_kwargs(chunk)

            program = self._program_factory(prompt_template)
            if resp := await self._aupdate_response(
                program, prompt_kwargs, response_kwargs
            ):
                response = resp

        if isinstance(response, str):
            if self._output_cls is not None:
                try:
                    response = self._output_cls.model_validate_json(response)
                except (ValidationError, ValueError, TypeError):
                    pass
            else:
                response = response or "Empty Response"
        elif response is None:
            response = "Empty Response"
        else:
            response = cast(AsyncGenerator, response)
        dispatcher.event(end_event)
        return response

    # TODO: Why does this class call dispatcher.span on this method when other classes only call it on synthesize
    @dispatcher.span
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response over chunks."""
        return self._run_refine_loop(
            query_str=query_str,
            chunks=text_chunks,
            qa_template=self._text_qa_template,
            refine_template=self._refine_template,
            make_qa_prompt_kwargs=lambda chunk: {"context_str": chunk},
            make_refine_prompt_kwargs=lambda chunk: {"context_msg": chunk},
            start_event=GetResponseStartEvent(
                query_str=query_str, text_chunks=list(text_chunks)
            ),
            end_event=GetResponseEndEvent(),
            prev_response=prev_response,
            **response_kwargs,
        )

    @dispatcher.span
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return await self._arun_refine_loop(
            query_str=query_str,
            chunks=text_chunks,
            qa_template=self._text_qa_template,
            refine_template=self._refine_template,
            make_qa_prompt_kwargs=lambda chunk: {"context_str": chunk},
            make_refine_prompt_kwargs=lambda chunk: {"context_msg": chunk},
            start_event=GetResponseStartEvent(
                query_str=query_str, text_chunks=list(text_chunks)
            ),
            end_event=GetResponseEndEvent(),
            prev_response=prev_response,
            **response_kwargs,
        )

    @dispatcher.span
    def get_response_from_messages(
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Give response over message chunks."""
        return self._run_refine_loop(
            query_str=query_str,
            chunks=message_chunks,
            qa_template=self._chat_content_qa_template,
            refine_template=self._chat_content_refine_template,
            make_qa_prompt_kwargs=lambda chunk: {"context_messages": [chunk]},
            make_refine_prompt_kwargs=lambda chunk: {"context_messages": [chunk]},
            start_event=GetMessageResponseStartEvent(
                query_str=query_str, message_chunks=list(message_chunks)
            ),
            end_event=GetMessageResponseEndEvent(),
            prev_response=prev_response,
            **response_kwargs,
        )

    @dispatcher.span
    async def aget_response_from_messages(
        self,
        query_str: str,
        message_chunks: Sequence[ChatMessage],
        prev_response: Optional[RESPONSE_TEXT_TYPE] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        return await self._arun_refine_loop(
            query_str=query_str,
            chunks=message_chunks,
            qa_template=self._chat_content_qa_template,
            refine_template=self._chat_content_refine_template,
            make_qa_prompt_kwargs=lambda chunk: {"context_messages": [chunk]},
            make_refine_prompt_kwargs=lambda chunk: {"context_messages": [chunk]},
            start_event=GetMessageResponseStartEvent(
                query_str=query_str, message_chunks=list(message_chunks)
            ),
            end_event=GetMessageResponseEndEvent(),
            prev_response=prev_response,
            **response_kwargs,
        )
