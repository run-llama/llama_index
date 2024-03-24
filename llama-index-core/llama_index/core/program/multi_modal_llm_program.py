from typing import Any, Dict, Optional, Sequence, Type, cast

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.schema import ImageDocument
from llama_index.core.types import BasePydanticProgram
from llama_index.core.utils import print_text


class MultiModalLLMCompletionProgram(BasePydanticProgram[BaseModel]):
    """
    Multi Modal LLM Completion Program.

    Uses generic Multi Modal LLM completion + an output parser to generate a structured output.

    """

    def __init__(
        self,
        output_parser: PydanticOutputParser,
        prompt: BasePromptTemplate,
        multi_modal_llm: MultiModalLLM,
        image_documents: Sequence[ImageDocument],
        verbose: bool = False,
    ) -> None:
        self._output_parser = output_parser
        self._multi_modal_llm = multi_modal_llm
        self._prompt = prompt
        self._image_documents = image_documents
        self._verbose = verbose

        self._prompt.output_parser = output_parser

    @classmethod
    def from_defaults(
        cls,
        output_parser: Optional[PydanticOutputParser] = None,
        output_cls: Optional[Type[BaseModel]] = None,
        prompt_template_str: Optional[str] = None,
        prompt: Optional[PromptTemplate] = None,
        multi_modal_llm: Optional[MultiModalLLM] = None,
        image_documents: Optional[Sequence[ImageDocument]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "MultiModalLLMCompletionProgram":
        if multi_modal_llm is None:
            try:
                from llama_index.multi_modal_llms.openai import (
                    OpenAIMultiModal,
                )  # pants: no-infer-dep

                multi_modal_llm = OpenAIMultiModal(
                    model="gpt-4-vision-preview", temperature=0
                )
            except ImportError as e:
                raise ImportError(
                    "`llama-index-multi-modal-llms-openai` package cannot be found. "
                    "Please install it by using `pip install `llama-index-multi-modal-llms-openai`"
                )
        if prompt is None and prompt_template_str is None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is not None and prompt_template_str is not None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt_template_str is not None:
            prompt = PromptTemplate(prompt_template_str)

        if output_parser is None:
            if output_cls is None:
                raise ValueError("Must provide either output_cls or output_parser.")
            output_parser = PydanticOutputParser(output_cls=output_cls)

        return cls(
            output_parser,
            prompt=cast(PromptTemplate, prompt),
            multi_modal_llm=multi_modal_llm,
            image_documents=image_documents or [],
            verbose=verbose,
        )

    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_parser.output_cls

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
    ) -> BaseModel:
        llm_kwargs = llm_kwargs or {}
        formatted_prompt = self._prompt.format(llm=self._multi_modal_llm, **kwargs)

        response = self._multi_modal_llm.complete(
            formatted_prompt,
            image_documents=self._image_documents,
            **llm_kwargs,
        )

        raw_output = response.text
        if self._verbose:
            print_text(f"> Raw output: {raw_output}\n", color="llama_blue")

        return self._output_parser.parse(raw_output)

    async def acall(
        self,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        llm_kwargs = llm_kwargs or {}
        formatted_prompt = self._prompt.format(llm=self._multi_modal_llm, **kwargs)

        response = await self._multi_modal_llm.acomplete(
            formatted_prompt,
            image_documents=self._image_documents,
            **llm_kwargs,
        )

        raw_output = response.text
        if self._verbose:
            print_text(f"> Raw output: {raw_output}\n", color="llama_blue")

        return self._output_parser.parse(raw_output)
