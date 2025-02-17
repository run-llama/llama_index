from typing import Any, Dict, List, Optional, Sequence, cast

from llama_index.core.base.base_selector import (
    BaseSelector,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.llms import LLM
from llama_index.core.output_parsers.base import StructuredOutput
from llama_index.core.output_parsers.selection import Answer, SelectionOutputParser
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.schema import QueryBundle
from llama_index.core.selectors.prompts import (
    DEFAULT_MULTI_SELECT_PROMPT_TMPL,
    DEFAULT_SINGLE_SELECT_PROMPT_TMPL,
    MultiSelectPrompt,
    SingleSelectPrompt,
)
from llama_index.core.settings import Settings
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.types import BaseOutputParser


def _build_choices_text(choices: Sequence[ToolMetadata]) -> str:
    """Convert sequence of metadata to enumeration text."""
    texts: List[str] = []
    for ind, choice in enumerate(choices):
        text = " ".join(choice.description.splitlines())
        text = f"({ind + 1}) {text}"  # to one indexing
        texts.append(text)
    return "\n\n".join(texts)


def _structured_output_to_selector_result(output: Any) -> SelectorResult:
    """Convert structured output to selector result."""
    structured_output = cast(StructuredOutput, output)
    answers = cast(List[Answer], structured_output.parsed_output)

    # adjust for zero indexing
    selections = [
        SingleSelection(index=answer.choice - 1, reason=answer.reason)
        for answer in answers
    ]
    return SelectorResult(selections=selections)


class LLMSingleSelector(BaseSelector):
    """LLM single selector.

    LLM-based selector that chooses one out of many options.

    Args:
        LLM (LLM): An LLM.
        prompt (SingleSelectPrompt): A LLM prompt for selecting one out of many options.
    """

    def __init__(
        self,
        llm: LLM,
        prompt: SingleSelectPrompt,
    ) -> None:
        self._llm = llm
        self._prompt: BasePromptTemplate = prompt

        if self._prompt.output_parser is None:
            raise ValueError("Prompt should have output parser.")

    @classmethod
    def from_defaults(
        cls,
        llm: Optional[LLM] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> "LLMSingleSelector":
        # optionally initialize defaults
        llm = llm or Settings.llm
        prompt_template_str = prompt_template_str or DEFAULT_SINGLE_SELECT_PROMPT_TMPL
        output_parser = output_parser or SelectionOutputParser()

        # construct prompt
        prompt = SingleSelectPrompt(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.SINGLE_SELECT,
        )
        return cls(llm, prompt)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {"prompt": self._prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "prompt" in prompts:
            self._prompt = prompts["prompt"]

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        choices_text = _build_choices_text(choices)

        # predict
        prediction = self._llm.predict(
            prompt=self._prompt,
            num_choices=len(choices),
            context_list=choices_text,
            query_str=query.query_str,
        )

        # parse output
        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        return _structured_output_to_selector_result(parse)

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        choices_text = _build_choices_text(choices)

        # predict
        prediction = await self._llm.apredict(
            prompt=self._prompt,
            num_choices=len(choices),
            context_list=choices_text,
            query_str=query.query_str,
        )

        # parse output
        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        return _structured_output_to_selector_result(parse)


class LLMMultiSelector(BaseSelector):
    """LLM multi selector.

    LLM-based selector that chooses multiple out of many options.

    Args:
        llm (LLM): An LLM.
        prompt (SingleSelectPrompt): A LLM prompt for selecting multiple out of many
            options.
    """

    def __init__(
        self,
        llm: LLM,
        prompt: MultiSelectPrompt,
        max_outputs: Optional[int] = None,
    ) -> None:
        self._llm = llm
        self._prompt: BasePromptTemplate = prompt
        self._max_outputs = max_outputs

        if self._prompt.output_parser is None:
            raise ValueError("Prompt should have output parser.")

    @classmethod
    def from_defaults(
        cls,
        llm: Optional[LLM] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
        max_outputs: Optional[int] = None,
    ) -> "LLMMultiSelector":
        llm = llm or Settings.llm
        prompt_template_str = prompt_template_str or DEFAULT_MULTI_SELECT_PROMPT_TMPL
        output_parser = output_parser or SelectionOutputParser()

        # add output formatting
        prompt_template_str = output_parser.format(prompt_template_str)

        # construct prompt
        prompt = MultiSelectPrompt(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.MULTI_SELECT,
        )
        return cls(llm, prompt, max_outputs)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {"prompt": self._prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "prompt" in prompts:
            self._prompt = prompts["prompt"]

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        context_list = _build_choices_text(choices)
        max_outputs = self._max_outputs or len(choices)

        prediction = self._llm.predict(
            prompt=self._prompt,
            num_choices=len(choices),
            max_outputs=max_outputs,
            context_list=context_list,
            query_str=query.query_str,
        )

        assert self._prompt.output_parser is not None
        parsed = self._prompt.output_parser.parse(prediction)
        return _structured_output_to_selector_result(parsed)

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        context_list = _build_choices_text(choices)
        max_outputs = self._max_outputs or len(choices)

        prediction = await self._llm.apredict(
            prompt=self._prompt,
            num_choices=len(choices),
            max_outputs=max_outputs,
            context_list=context_list,
            query_str=query.query_str,
        )

        assert self._prompt.output_parser is not None
        parsed = self._prompt.output_parser.parse(prediction)
        return _structured_output_to_selector_result(parsed)
