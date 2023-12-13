from typing import TYPE_CHECKING, List, Optional, Sequence, cast

from llama_index.program.guidance_program import GuidancePydanticProgram
from llama_index.prompts.guidance_utils import convert_to_handlebars
from llama_index.prompts.mixin import PromptDictType
from llama_index.question_gen.prompts import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL,
    build_tools_text,
)
from llama_index.question_gen.types import (
    BaseQuestionGenerator,
    SubQuestion,
    SubQuestionList,
)
from llama_index.schema import QueryBundle
from llama_index.tools.types import ToolMetadata

if TYPE_CHECKING:
    from guidance.llms import LLM as GuidanceLLM

DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL = convert_to_handlebars(
    DEFAULT_SUB_QUESTION_PROMPT_TMPL
)


class GuidanceQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        program: GuidancePydanticProgram,
        verbose: bool = False,
    ) -> None:
        self._program = program
        self._verbose = verbose

    @classmethod
    def from_defaults(
        cls,
        prompt_template_str: str = DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL,
        guidance_llm: Optional["GuidanceLLM"] = None,
        verbose: bool = False,
    ) -> "GuidanceQuestionGenerator":
        program = GuidancePydanticProgram(
            output_cls=SubQuestionList,
            guidance_llm=guidance_llm,
            prompt_template_str=prompt_template_str,
            verbose=verbose,
        )

        return cls(program, verbose)

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        question_list = self._program(
            tools_str=tools_str,
            query_str=query_str,
        )
        question_list = cast(SubQuestionList, question_list)
        return question_list.items

    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        # TODO: currently guidance does not support async calls
        return self.generate(tools=tools, query=query)
