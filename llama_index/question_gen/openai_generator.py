from typing import List, Optional, Sequence, cast

from llama_index.indices.query.schema import QueryBundle
from llama_index.llms.base import LLM
from llama_index.llms.openai import OpenAI
from llama_index.program.openai_program import OpenAIPydanticProgram
from llama_index.prompts.mixin import PromptDictType
from llama_index.question_gen.prompts import build_tools_text
from llama_index.question_gen.types import (
    BaseQuestionGenerator,
    SubQuestion,
    SubQuestionList,
)
from llama_index.tools.types import ToolMetadata

DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"

DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL = """\
You are a world class state of the art agent.

You have access to multiple tools, each representing a different data source or API.
Each of the tools has a name and a description, formatted as a JSON dictionary.
The keys of the dictionary are the names of the tools and the values are the \
descriptions.
Your purpose is to help answer a complex user question by generating a list of sub \
questions that can be answered by the tools.

These are the guidelines you consider when completing your task:
* Be as specific as possible
* The sub questions should be relevant to the user question
* The sub questions should be answerable by the tools provided
* You can generate multiple sub questions for each tool
* Tools must be specified by their name, not their description
* You don't need to use a tool if you don't think it's relevant

Output the list of sub questions by calling the SubQuestionList function.

## Tools
```json
{tools_str}
```

## User Question
{query_str}
"""


class OpenAIQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        program: OpenAIPydanticProgram,
        verbose: bool = False,
    ) -> None:
        self._program = program
        self._verbose = verbose

    @classmethod
    def from_defaults(
        cls,
        prompt_template_str: str = DEFAULT_OPENAI_SUB_QUESTION_PROMPT_TMPL,
        llm: Optional[LLM] = None,
        verbose: bool = False,
    ) -> "OpenAIQuestionGenerator":
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        program = OpenAIPydanticProgram.from_defaults(
            output_cls=SubQuestionList,
            llm=llm,
            prompt_template_str=prompt_template_str,
            verbose=verbose,
        )
        return cls(program, verbose)

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"question_gen_prompt": self._program.prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "question_gen_prompt" in prompts:
            self._program.prompt = prompts["question_gen_prompt"]

    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        question_list = self._program(query_str=query_str, tools_str=tools_str)
        question_list = cast(SubQuestionList, question_list)
        return question_list.items

    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        question_list = await self._program.acall(
            query_str=query_str, tools_str=tools_str
        )
        question_list = cast(SubQuestionList, question_list)
        return question_list.items
