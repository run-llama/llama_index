from typing import Any

from llama_index.output_parsers.base import BaseOutputParser, StructuredOutput
from llama_index.output_parsers.utils import parse_json_markdown
from llama_index.question_gen.types import SubQuestion


class SubQuestionOutputParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        json_dict = parse_json_markdown(output)
        sub_questions = [SubQuestion.parse_obj(item) for item in json_dict]
        return StructuredOutput(raw_output=output, parsed_output=sub_questions)

    def format(self, prompt_template: str) -> str:
        del prompt_template
        raise NotImplementedError()
