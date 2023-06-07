from typing import List, Optional, Sequence

from guidance import Program
from guidance.llms import LLM, OpenAI

from llama_index.indices.query.schema import QueryBundle
from llama_index.question_gen.prompts import (DEFAULT_SUB_QUESTION_PROMPT_TMPL,
                                              build_tools_text)
from llama_index.question_gen.types import BaseQuestionGenerator, SubQuestion
from llama_index.tools.types import ToolMetadata


def _convert_fstring_to_guidance_template(prompt_template: str) -> str:
    pass


DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL = _convert_fstring_to_guidance_template(
    DEFAULT_SUB_QUESTION_PROMPT_TMPL
)


class GuidanceQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        guidance_program: Program,
    ) -> None:
        # construct guidance program
        self._guidance_program = guidance_program

    @classmethod
    def from_defaults(
        cls,
        prompt_template_str: str = DEFAULT_GUIDANCE_SUB_QUESTION_PROMPT_TMPL,
        llm: Optional[LLM] = None,
    ):
        llm = llm or OpenAI("text-davinci-003")
        guidance_program = Program(prompt_template_str, llm=llm)
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

        # TODO: figure out how to get this from the program
        output = program.get()
        return output

    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        pass
