from typing import List, Optional, Sequence, cast

from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers.base import StructuredOutput
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.question_gen.output_parser import SubQuestionOutputParser
from llama_index.core.question_gen.prompts import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL,
    build_tools_text,
)
from llama_index.core.question_gen.types import BaseQuestionGenerator, SubQuestion
from llama_index.core.schema import QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.types import BaseOutputParser


class LLMQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        llm: LLM,
        prompt: BasePromptTemplate,
    ) -> None:
        self._llm = llm
        self._prompt = prompt

        if self._prompt.output_parser is None:
            raise ValueError("Prompt should have output parser.")

    @classmethod
    def from_defaults(
        cls,
        llm: Optional[LLM] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> "LLMQuestionGenerator":
        # optionally initialize defaults
        llm = llm or Settings.llm
        prompt_template_str = prompt_template_str or DEFAULT_SUB_QUESTION_PROMPT_TMPL
        output_parser = output_parser or SubQuestionOutputParser()

        # construct prompt
        prompt = PromptTemplate(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.SUB_QUESTION,
        )
        return cls(llm, prompt)

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"question_gen_prompt": self._prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "question_gen_prompt" in prompts:
            output_parser = prompts["question_gen_prompt"].output_parser
            if output_parser is None:
                output_parser = SubQuestionOutputParser()
            self._prompt = PromptTemplate(
                prompts["question_gen_prompt"].get_template(llm=self._llm),
                output_parser=output_parser,
            )

    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = self._llm.predict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output

    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = await self._llm.apredict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output
