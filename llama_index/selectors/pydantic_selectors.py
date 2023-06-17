from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel
from typing import Any, List, Optional, Sequence

from llama_index.indices.query.schema import QueryBundle
from llama_index.program.openai_program import (
    OpenAIPydanticProgram,
    BasePydanticProgram,
    Model,
)
from llama_index.selectors.llm_selectors import _build_choices_text
from llama_index.selectors.prompts import (
    DEFAULT_MULTI_PYD_SELECT_PROMPT_TMPL,
    DEFAULT_SINGLE_PYD_SELECT_PROMPT_TMPL,
)
from llama_index.selectors.types import BaseSelector, SelectorResult
from llama_index.tools.types import ToolMetadata


class SingleSelection(Model):
    index: int
    reason: str


class MultiSelection(Model):
    selections: List[SingleSelection]


def _pydantic_output_to_selector_result(output: Any) -> SelectorResult:
    """
    Convert pydantic output to selector result.
    Takes into account zero-indexing on answer indexes.
    """
    if isinstance(output, SingleSelection):
        return SelectorResult(inds=[output.index - 1], reasons=[output.reason])
    elif isinstance(output, MultiSelection):
        return SelectorResult(
            inds=[x.index - 1 for x in output.selections],
            reasons=[x.reason for x in output.selections],
        )
    else:
        raise ValueError(f"Unsupported output type: {type(output)}")


class PydanticSingleSelector(BaseSelector):
    def __init__(self, selector_program: BasePydanticProgram) -> None:
        self._selector_program = selector_program

    @classmethod
    def from_defaults(
        cls,
        program: Optional[BasePydanticProgram] = None,
        llm: Optional[ChatOpenAI] = None,
        output_cls: Optional[Model] = None,
        prompt_template_str: str = DEFAULT_SINGLE_PYD_SELECT_PROMPT_TMPL,
        verbose: bool = False,
    ) -> "PydanticSingleSelector":

        if program is None:
            output_cls = output_cls or SingleSelection
            program = OpenAIPydanticProgram.from_defaults(
                output_cls=output_cls,
                prompt_template_str=prompt_template_str,
                llm=llm,
                verbose=verbose,
            )

        return cls(selector_program=program)

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        choices_text = _build_choices_text(choices)

        # predict
        prediction = self._selector_program(
            num_choices=len(choices),
            context_list=choices_text,
            query_str=query.query_str,
        )

        # parse output
        return _pydantic_output_to_selector_result(prediction)

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        raise NotImplementedError(
            "Async selection not supported for Pydantic Selectors."
        )


class PydanticMultiSelector(BaseSelector):
    def __init__(
        self, selector_program: BasePydanticProgram, max_outputs: Optional[int] = None
    ) -> None:
        self._selector_program = selector_program
        self._max_outputs = max_outputs

    @classmethod
    def from_defaults(
        cls,
        program: Optional[BasePydanticProgram] = None,
        llm: Optional[ChatOpenAI] = None,
        output_cls: Optional[BaseModel] = None,
        prompt_template_str: str = DEFAULT_MULTI_PYD_SELECT_PROMPT_TMPL,
        max_outputs: Optional[int] = None,
        verbose: bool = False,
    ) -> "PydanticMultiSelector":

        if program is None:
            output_cls = output_cls or MultiSelection
            program = OpenAIPydanticProgram.from_defaults(
                output_cls=output_cls,
                prompt_template_str=prompt_template_str,
                llm=llm,
                verbose=verbose,
            )

        return cls(selector_program=program, max_outputs=max_outputs)

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        context_list = _build_choices_text(choices)
        max_outputs = self._max_outputs or len(choices)

        # predict
        prediction = self._selector_program(
            num_choices=len(choices),
            max_outputs=max_outputs,
            context_list=context_list,
            query_str=query.query_str,
        )

        # parse output
        return _pydantic_output_to_selector_result(prediction)

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        raise NotImplementedError(
            "Async selection not supported for Pydantic Selectors."
        )
