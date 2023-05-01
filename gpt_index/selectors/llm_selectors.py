from typing import List, Optional, Sequence, cast
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.service_context import ServiceContext


from gpt_index.llm_predictor.base import LLMPredictor
from gpt_index.output_parsers.base import BaseOutputParser, StructuredOutput
from gpt_index.output_parsers.selection import (
    ANSWERS_KEY,
    Answer,
    SelectionOutputParser,
)
from gpt_index.selectors.prompts import (
    DEFAULT_MULTI_SELECT_PROMPT_TMPL,
    DEFAULT_SINGLE_SELECT_PROMPT_TMPL,
    MultiSelectPrompt,
    SingleSelectPrompt,
)
from gpt_index.selectors.types import BaseSelector, Metadata, SelectorResult


def _build_choices_text(choices: Sequence[Metadata]) -> str:
    texts = []
    for ind, choice in enumerate(choices):
        text = " ".join(choice.description.splitlines())
        text = f"({ind + 1}) {text}"
    return "\n\n".join(texts)


def _structured_output_to_selector_result(output: StructuredOutput) -> SelectorResult:
    answers = output.parsed_output[ANSWERS_KEY]
    answers = cast(List[Answer], answers)
    inds = [answer.choice - 1 for answer in answers]  # for zero indexing
    reasons = [answer.reason for answer in answers]
    return SelectorResult(inds=inds, reasons=reasons)


class LLMSingleSelector(BaseSelector):
    def __init__(
        self,
        llm_predictor: LLMPredictor,
        prompt: SingleSelectPrompt,
    ) -> None:
        self._llm_predictor = llm_predictor
        self._prompt = prompt

        if self._prompt.output_parser is None:
            raise ValueError("Prompt should have output parser.")

    @classmethod
    def from_defaults(
        cls,
        service_context: Optional[ServiceContext] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> "LLMSingleSelector":
        service_context = service_context or ServiceContext.from_defaults()
        prompt_template_str = prompt_template_str or DEFAULT_SINGLE_SELECT_PROMPT_TMPL
        output_parser = output_parser or SelectionOutputParser()

        # add output formatting
        prompt_template_str = output_parser.format(prompt_template_str)

        # construct prompt
        prompt = SingleSelectPrompt(
            template=prompt_template_str, output_parser=output_parser
        )
        return cls(service_context.llm_predictor, prompt)

    def _select(
        self, choices: Sequence[Metadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        choices_text = _build_choices_text(choices)

        # predict
        prediction, _ = self._llm_predictor.predict(
            prompt=self._prompt,
            num_choices=len(choices),
            context_list=choices_text,
            query_str=query.query_str,
        )

        # parse output
        parse = self._prompt.output_parser.parse(prediction)
        assert isinstance(parse, StructuredOutput)
        return _structured_output_to_selector_result(parse)

    async def _aselect(
        self, choices: Sequence[Metadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        choices_text = _build_choices_text(choices)

        # predict
        prediction, _ = self._llm_predictor.apredict(
            prompt=self._prompt,
            num_choices=len(choices),
            context_list=choices_text,
            query_str=query.query_str,
        )

        # parse output
        parse = self._prompt.output_parser.parse(prediction)
        assert isinstance(parse, StructuredOutput)
        return _structured_output_to_selector_result(parse)


class LLMMultiSelector(BaseSelector):
    def __init__(
        self,
        llm_predictor: LLMPredictor,
        prompt: MultiSelectPrompt,
        max_choices: Optional[int] = None,
    ) -> None:
        self._llm_predictor = llm_predictor
        self._prompt = prompt
        self._max_choices = max_choices

        if self._prompt.output_parser is None:
            raise ValueError("Prompt should have output parser.")

    @classmethod
    def from_defaults(
        cls,
        service_context: Optional[ServiceContext] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
        max_choices: Optional[int] = None,
    ) -> "LLMMultiSelector":
        service_context = service_context or ServiceContext.from_defaults()
        prompt_template_str = prompt_template_str or DEFAULT_MULTI_SELECT_PROMPT_TMPL
        output_parser = output_parser or SelectionOutputParser()

        # add output formatting
        prompt_template_str = output_parser.format(prompt_template_str)

        # construct prompt
        prompt = MultiSelectPrompt(
            template=prompt_template_str, output_parser=output_parser
        )
        return cls(service_context.llm_predictor, prompt, max_choices)

    def _select(
        self, choices: Sequence[Metadata], query: QueryBundle
    ) -> SelectorResult:

        context_list = _build_choices_text(choices)

        max_choices = self._max_choices or len(choices)
        prediction, _ = self._llm_predictor.predict(
            prompt=self._prompt,
            num_choices=len(choices),
            max_choices=max_choices,
            context_list=context_list,
            query_str=query.query_str,
        )

        parse = self._prompt.output_parser.parse(prediction)
        assert isinstance(parse, StructuredOutput)
        return _structured_output_to_selector_result(parse)

    async def _aselect(
        self, choices: Sequence[Metadata], query: QueryBundle
    ) -> SelectorResult:

        context_list = _build_choices_text(choices)

        max_choices = self._max_choices or len(choices)
        prediction, _ = self._llm_predictor.predict(
            prompt=self._prompt,
            num_choices=len(choices),
            max_choices=max_choices,
            context_list=context_list,
            query_str=query.query_str,
        )

        parse = self._prompt.output_parser.parse(prediction)
        assert isinstance(parse, StructuredOutput)
        return _structured_output_to_selector_result(parse)
