from string import Formatter
from typing import List

from llama_index.llms.base import LLM


def get_template_vars(template_str: str) -> List[str]:
    """Get template variables from a template string."""
    variables = []
    formatter = Formatter()

    for _, variable_name, _, _ in formatter.parse(template_str):
        if variable_name:
            variables.append(variable_name)

    return variables


def is_chat_model(llm: LLM) -> bool:
    return llm.metadata.is_chat_model
