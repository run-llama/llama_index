from typing import TYPE_CHECKING, List, Optional, Sequence, cast

from pydantic import BaseModel

from llama_index.prompts.guidance_utils import (
    convert_to_handlebars, parse_pydantic_from_guidance_program,
    pydantic_to_guidance_output_template_markdown)

if TYPE_CHECKING:
    from guidance import Program
    from guidance.llms import LLM

from llama_index.indices.query.schema import QueryBundle
from llama_index.question_gen.prompts import (DEFAULT_SUB_QUESTION_PROMPT_TMPL,
                                              build_tools_text)
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
        guidance_program: "Program",
    ) -> None:
        self._guidance_program = guidance_program

    @classmethod
    def from_defaults(
        cls,
        prompt_template_str: str = DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL,
        output_str: Optional[str] = None,
        llm: Optional["LLM"] = None,
    ) -> "GuidanceQuestionGenerator":
        try:
            from guidance import Program
            from guidance.llms import OpenAI
        except ImportError as e:
            raise ImportError(
                "guidance package not found." "please run `pip install guidance`"
            )

        # construct guidance program
        llm = llm or OpenAI("text-davinci-003")
        output_str = output_str or pydantic_to_guidance_output_template_markdown(
            SubQuestionList
        )
        full_str = prompt_template_str + "\n" + output_str
        guidance_program = Program(full_str, llm=llm, silent=True)

        return cls(guidance_program)

    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        program = self._guidance_program(
            tools_str=tools_str,
            query_str=query_str,
        )

        subq_list = parse_pydantic_from_guidance_program(
            program=program, cls=SubQuestionList
        )
        return subq_list.sub_questions

    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        # TODO: currently guidance does not support async calls
        return self.generate(tools=tools, query=query)
