from typing import Any

from llama_index.legacy.output_parsers.base import StructuredOutput
from llama_index.legacy.output_parsers.utils import parse_json_markdown
from llama_index.legacy.question_gen.types import SubQuestion
from llama_index.legacy.types import BaseOutputParser


class SubQuestionOutputParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        json_dict = parse_json_markdown(output)
        if not json_dict:
            raise ValueError(f"No valid JSON found in output: {output}")

        # example code includes an 'items' key, which breaks
        # the parsing from open-source LLMs such as Zephyr.
        # This gets the actual subquestions and recommended tools directly
        if "items" in json_dict:
            json_dict = json_dict["items"]

        sub_questions = [SubQuestion.parse_obj(item) for item in json_dict]
        return StructuredOutput(raw_output=output, parsed_output=sub_questions)

    def format(self, prompt_template: str) -> str:
        return prompt_template
