from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from llama_index.core.base.base_selector import (
    BaseSelector,
    MultiSelection,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle
from llama_index.core.selectors.llm_selectors import _build_choices_text
from llama_index.core.selectors.prompts import (
    DEFAULT_MULTI_PYD_SELECT_PROMPT_TMPL,
    DEFAULT_SINGLE_PYD_SELECT_PROMPT_TMPL,
)
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.types import BasePydanticProgram

if TYPE_CHECKING:
    from llama_index.llms.openai import OpenAI  # pants: no-infer-dep


def _pydantic_output_to_selector_result(output: Any) -> SelectorResult:
    """
    Convert pydantic output to selector result.
    Takes into account zero-indexing on answer indexes.
    """
    if isinstance(output, SingleSelection):
        output.index -= 1
        return SelectorResult(selections=[output])
    elif isinstance(output, MultiSelection):
        for idx in range(len(output.selections)):
            output.selections[idx].index -= 1
        return SelectorResult(selections=output.selections)
    else:
        raise ValueError(f"Unsupported output type: {type(output)}")


class PydanticSingleSelector(BaseSelector):
    def __init__(self, selector_program: BasePydanticProgram) -> None:
        self._selector_program = selector_program

    @classmethod
    def from_defaults(
        cls,
        program: Optional[BasePydanticProgram] = None,
        llm: Optional["OpenAI"] = None,
        prompt_template_str: str = DEFAULT_SINGLE_PYD_SELECT_PROMPT_TMPL,
        verbose: bool = False,
    ) -> "PydanticSingleSelector":
        try:
            from llama_index.program.openai import (
                OpenAIPydanticProgram,
            )  # pants: no-infer-dep
        except ImportError as e:
            raise ImportError(
                "`llama-index-program-openai` package is missing. "
                "Please install using `pip install llama-index-program-openai`."
            )
        if program is None:
            program = OpenAIPydanticProgram.from_defaults(
                output_cls=SingleSelection,
                prompt_template_str=prompt_template_str,
                llm=llm,
                verbose=verbose,
            )

        return cls(selector_program=program)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        # TODO: no accessible prompts for a base pydantic program
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

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
        llm: Optional["OpenAI"] = None,
        prompt_template_str: str = DEFAULT_MULTI_PYD_SELECT_PROMPT_TMPL,
        max_outputs: Optional[int] = None,
        verbose: bool = False,
    ) -> "PydanticMultiSelector":
        try:
            from llama_index.program.openai import (
                OpenAIPydanticProgram,
            )  # pants: no-infer-dep
        except ImportError as e:
            raise ImportError(
                "`llama-index-program-openai` package is missing. "
                "Please install using `pip install llama-index-program-openai`."
            )
        if program is None:
            program = OpenAIPydanticProgram.from_defaults(
                output_cls=MultiSelection,
                prompt_template_str=prompt_template_str,
                llm=llm,
                verbose=verbose,
            )

        return cls(selector_program=program, max_outputs=max_outputs)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        # TODO: no accessible prompts for a base pydantic program
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

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
        return self._select(choices, query)
