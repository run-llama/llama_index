import logging
from typing import (
    Any,
    Dict,
    Generator,
    Optional,
    Type,
    cast,
    AsyncGenerator,
    Union,
    List,
)

from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.program.utils import FlexibleModel, process_streaming_objects
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.types import BaseOutputParser, BasePydanticProgram, Model


_logger = logging.getLogger(__name__)


class LLMTextCompletionProgram(BasePydanticProgram[Model]):
    """
    LLM Text Completion Program.

    Uses generic LLM text completion + an output parser to generate a structured output.

    """

    def __init__(
        self,
        output_parser: BaseOutputParser,
        output_cls: Type[Model],
        prompt: BasePromptTemplate,
        llm: LLM,
        verbose: bool = False,
    ) -> None:
        self._output_parser = output_parser
        self._output_cls = output_cls
        self._llm = llm
        self._prompt = prompt
        self._verbose = verbose

        self._prompt.output_parser = output_parser

    @classmethod
    def from_defaults(
        cls,
        output_parser: Optional[BaseOutputParser] = None,
        output_cls: Optional[Type[Model]] = None,
        prompt_template_str: Optional[str] = None,
        prompt: Optional[BasePromptTemplate] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "LLMTextCompletionProgram[Model]":
        llm = llm or Settings.llm
        if prompt is None and prompt_template_str is None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is not None and prompt_template_str is not None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt_template_str is not None:
            prompt = PromptTemplate(prompt_template_str)

        # decide default output class if not set
        if output_cls is None:
            if not isinstance(output_parser, PydanticOutputParser):
                raise ValueError("Output parser must be PydanticOutputParser.")
            output_cls = output_parser.output_cls
        else:
            if output_parser is None:
                output_parser = PydanticOutputParser(output_cls=output_cls)

        return cls(
            output_parser,
            output_cls,
            prompt=cast(PromptTemplate, prompt),
            llm=llm,
            verbose=verbose,
        )

    @property
    def output_cls(self) -> Type[Model]:
        return self._output_cls

    @property
    def prompt(self) -> BasePromptTemplate:
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: BasePromptTemplate) -> None:
        self._prompt = prompt

    def __call__(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Model:
        llm_kwargs = llm_kwargs or {}
        if self._llm.metadata.is_chat_model:
            messages = self._prompt.format_messages(llm=self._llm, **kwargs)
            messages = self._llm._extend_messages(messages)
            chat_response = self._llm.chat(messages, **llm_kwargs)

            raw_output = chat_response.message.content or ""
        else:
            formatted_prompt = self._prompt.format(llm=self._llm, **kwargs)

            response = self._llm.complete(formatted_prompt, **llm_kwargs)

            raw_output = response.text

        output = self._output_parser.parse(raw_output)
        if not isinstance(output, self._output_cls):
            raise ValueError(
                f"Output parser returned {type(output)} but expected {self._output_cls}"
            )
        return output

    async def acall(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Model:
        llm_kwargs = llm_kwargs or {}
        if self._llm.metadata.is_chat_model:
            messages = self._prompt.format_messages(llm=self._llm, **kwargs)
            messages = self._llm._extend_messages(messages)
            chat_response = await self._llm.achat(messages, **llm_kwargs)

            raw_output = chat_response.message.content or ""
        else:
            formatted_prompt = self._prompt.format(llm=self._llm, **kwargs)

            response = await self._llm.acomplete(formatted_prompt, **llm_kwargs)

            raw_output = response.text

        output = self._output_parser.parse(raw_output)
        if not isinstance(output, self._output_cls):
            raise ValueError(
                f"Output parser returned {type(output)} but expected {self._output_cls}"
            )
        return output

    def stream_call(
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Generator[
        Union[Model, List[Model], FlexibleModel, List[FlexibleModel]], None, None
    ]:
        """
        Stream object.

        Returns a generator returning partials of the same object
        or a list of objects until it returns.
        """
        llm_kwargs = llm_kwargs or {}
        if self._llm.metadata.is_chat_model:
            messages = self._prompt.format_messages(llm=self._llm, **kwargs)
            messages = self._llm._extend_messages(messages)
            response_gen = self._llm.stream_chat(messages, **llm_kwargs)
        else:
            formatted_prompt = self._prompt.format(llm=self._llm, **kwargs)
            response_gen = self._llm.stream_complete(formatted_prompt, **llm_kwargs)  # type: ignore[assignment]
        cur_objects = None
        for partial_resp in response_gen:
            try:
                objects = process_streaming_objects(
                    partial_resp,
                    self._output_cls,
                    cur_objects=cur_objects,
                    flexible_mode=True,
                    llm=self._llm,  # type: ignore[arg-type]
                )
                cur_objects = objects if isinstance(objects, list) else [objects]
                yield objects
            except Exception as e:
                _logger.warning(f"Failed to parse streaming response: {e}")
                continue

    async def astream_call(
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> AsyncGenerator[
        Union[Model, List[Model], FlexibleModel, List[FlexibleModel]], None
    ]:
        """
        Stream objects.

        Returns a generator returning partials of the same object
        or a list of objects until it returns.
        """
        llm_kwargs = llm_kwargs or {}
        if self._llm.metadata.is_chat_model:
            messages = self._prompt.format_messages(llm=self._llm, **kwargs)
            messages = self._llm._extend_messages(messages)
            response_gen = await self._llm.astream_chat(messages, **llm_kwargs)
        else:
            formatted_prompt = self._prompt.format(llm=self._llm, **kwargs)
            response_gen = await self._llm.astream_complete(  # type: ignore[assignment]
                formatted_prompt, **llm_kwargs
            )

        async def gen() -> AsyncGenerator[
            Union[Model, List[Model], FlexibleModel, List[FlexibleModel]], None
        ]:
            cur_objects = None
            async for partial_resp in response_gen:
                try:
                    objects = process_streaming_objects(
                        partial_resp,
                        self._output_cls,
                        cur_objects=cur_objects,
                        flexible_mode=True,
                        llm=self._llm,  # type: ignore[arg-type]
                    )
                    cur_objects = objects if isinstance(objects, list) else [objects]
                    yield objects
                except Exception as e:
                    _logger.warning(f"Failed to parse streaming response: {e}")
                    continue

        return gen()
