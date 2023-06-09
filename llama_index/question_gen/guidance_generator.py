from typing import TYPE_CHECKING, List, Optional, Sequence

from pydantic import BaseModel

from llama_index.program.guidance_program import PydanticGuidanceProgram
from llama_index.prompts.guidance_utils import (
    convert_to_handlebars,
    parse_pydantic_from_guidance_program,
    pydantic_to_guidance_output_template_markdown,
)

if TYPE_CHECKING:
    from guidance import Program
    from guidance.llms import LLM

from llama_index.indices.query.schema import QueryBundle
from llama_index.question_gen.prompts import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL,
    build_tools_text,
)
from llama_index.question_gen.types import BaseQuestionGenerator, SubQuestion
from llama_index.tools.types import ToolMetadata

DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL = convert_to_handlebars(
    DEFAULT_SUB_QUESTION_PROMPT_TMPL
)


class SubQuestionList(BaseModel):
    sub_questions: List[SubQuestion]


class GuidanceQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        program: PydanticGuidanceProgram,
        verbose: bool = False,
    ) -> None:
        self._program = program
        self._verbose = verbose

    @classmethod
    def from_defaults(
        cls,
        prompt_template_str: str = DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL,
        llm: Optional["LLM"] = None,
        verbose: bool = False,
    ) -> "GuidanceQuestionGenerator":
        program = PydanticGuidanceProgram(
            output_cls=SubQuestionList,
            llm=llm,
            prompt_template_str=prompt_template_str,
            verbose=verbose,
        )

        return cls(program, verbose)

    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        return self._program(
            tools_str=tools_str,
            query_str=query_str,
        )

    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        # TODO: currently guidance does not support async calls
        return self.generate(tools=tools, query=query)
